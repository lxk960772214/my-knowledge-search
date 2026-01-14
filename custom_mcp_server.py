from mcp.server.fastmcp import FastMCP
import os
import sys
import argparse
from typing import List, Dict, Tuple
import json
import requests
import numpy as np

# RAG 依赖 (延迟加载以避免启动慢)
try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: RAG dependencies (langchain) not found. Falling back to simple mode.", file=sys.stderr)

# 解析命令行参数
parser = argparse.ArgumentParser(description='Local Knowledge MCP Server')
parser.add_argument('roots', metavar='DIR', type=str, nargs='+',
                    help='Directories to search in')

# 初始化 MCP Server
mcp = FastMCP("LocalKnowledgeSearch")

# 获取搜索目录
SEARCH_ROOTS = []

def _init_search_roots():
    global SEARCH_ROOTS
    args = sys.argv[1:]
    for arg in args:
        if os.path.isdir(arg):
            SEARCH_ROOTS.append(os.path.abspath(arg))
    
    if not SEARCH_ROOTS:
         SEARCH_ROOTS = []
    print(f"Initialized search roots: {SEARCH_ROOTS}", file=sys.stderr)

_init_search_roots()

import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# --- Semantic RAG Core ---
class LocalRAG:
    def __init__(self):
        if not RAG_AVAILABLE:
            return
        # Cloud Embedding Configuration
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        self.api_key = "sk-or-v1-d1a3f5ee13d0fd4627*************2790a79c77ddcdc5ca694c9048a01f"
        self.model_name = "baai/bge-m3"
        
        # Configure requests with retry logic
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        print("RAG initialized with Cloud Embedding (DeepInfra/BAAI).", file=sys.stderr)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Call Cloud API to get embeddings for a list of texts"""
        try:
            payload = {
                "model": self.model_name,
                "input": texts,
                "encoding_format": "float"
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Use session with retry logic
            response = self.session.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            # Extract embeddings ensuring correct order by index
            embeddings_map = {item['index']: item['embedding'] for item in data['data']}
            ordered_embeddings = [embeddings_map[i] for i in range(len(texts))]
            
            return np.array(ordered_embeddings)
            
        except Exception as e:
            print(f"Embedding API Error: {e}", file=sys.stderr)
            raise e

    def split_markdown(self, content: str) -> List[str]:
        """使用成熟的 LangChain 分块器按标题层级切分"""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        docs = splitter.split_text(content)
        # 将分割后的 Document 对象转回带上下文的字符串
        chunks = []
        for doc in docs:
            # 组合元数据和内容，保留上下文
            header_context = " > ".join(filter(None, [
                doc.metadata.get("Header 1"),
                doc.metadata.get("Header 2"),
                doc.metadata.get("Header 3")
            ]))
            text = f"Context: {header_context}\nContent: {doc.page_content}"
            chunks.append(text)
        return chunks

    def compute_similarity(self, query_vec, doc_vecs):
        """计算余弦相似度"""
        norm_query = np.linalg.norm(query_vec)
        norm_docs = np.linalg.norm(doc_vecs, axis=1)
        
        # 避免除零
        if norm_query == 0: return np.zeros(len(doc_vecs))
        norm_docs[norm_docs == 0] = 1
        
        return np.dot(doc_vecs, query_vec) / (norm_docs * norm_query)

# 全局 RAG 实例 (懒加载)
_rag_instance = None
def get_rag():
    global _rag_instance
    if _rag_instance is None and RAG_AVAILABLE:
        _rag_instance = LocalRAG()
    return _rag_instance

@mcp.tool()
def search_files(query: str) -> str:
    """
    Search for files using fuzzy keywords. 
    Matches filenames and YAML Front Matter metadata (tags, aliases, clients) with high priority.
    PERFORMANCE MODE: Skips full content search.
    
    Args:
        query: The search term (e.g., "python", "byd", "migration").
    """
    results = []
    query_lower = query.lower()
    
    print(f"Searching for '{query}' in {len(SEARCH_ROOTS)} directories (Performance Mode)...")

    for root_dir in SEARCH_ROOTS:
        if not os.path.exists(root_dir):
            continue
            
        # Search in current directory
        for root, dirs, files in os.walk(root_dir):
            # Check if current directory name matches query
            current_dir_name = os.path.basename(root)
            if query_lower in current_dir_name.lower():
                # If directory matches, prioritize readme*.md inside it
                for file in files:
                    if file.lower().startswith('readme') and file.lower().endswith('.md'):
                         results.append({
                            "score": 50, # High score for directory match
                            "name": file,
                            "path": os.path.join(root, file),
                            "reason": f"DirectoryMatch({current_dir_name})"
                        })

            for file in files:
                # 排除系统文件和非MD文件
                if file.startswith('.') or not file.lower().endswith('.md'):
                    continue
                
                file_path = os.path.join(root, file)
                score = 0
                match_reasons = []
                
                # 1. 文件名匹配
                if query_lower in file.lower():
                    score += 10
                    match_reasons.append("Filename")
                
                # [NEW] 1.5 目录名匹配 (Customer Directory Awareness)
                # 如果当前文件是 readme.md 且其父目录名包含查询词
                if file.lower().startswith('readme'):
                    parent_dir_name = os.path.basename(root)
                    if query_lower in parent_dir_name.lower():
                        score += 20  # 极高权重
                        match_reasons.append(f"ParentDir({parent_dir_name})")
                
                # 2. 仅匹配 YAML Front Matter (文件头)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        head_content = f.read(500) 
                        if head_content.startswith('---'):
                            end_idx = head_content.find('---', 3)
                            if end_idx != -1:
                                yaml_block = head_content[3:end_idx].lower()
                                if query_lower in yaml_block:
                                    score += 15
                                    match_reasons.append("Metadata/Tag")
                except Exception:
                    pass
                
                if score > 0:
                    results.append({
                        "path": file_path,
                        "name": file,
                        "reason": ", ".join(match_reasons),
                        "score": score
                    })

    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Deduplicate results based on file path (Keep the highest score)
    unique_results = []
    seen_paths = set()
    for res in results:
        if res['path'] not in seen_paths:
            unique_results.append(res)
            seen_paths.add(res['path'])
    results = unique_results
    
    if not results:
        return f"No files found matching '{query}'."
    
    output = [f"Found {len(results)} matches for '{query}':\n"]
    for res in results[:5]:
        output.append(f"- [{res['reason']}] {res['name']} (Score: {res['score']})")
        output.append(f"  Path: {res['path']}")
    
    return "\n".join(output)

@mcp.tool()
def read_relevant_content(file_path: str, query: str = None) -> str:
    """
    Read specific parts of a file relevant to a query using Semantic RAG.
    Uses 'fastembed' for vector similarity and 'langchain' for smart markdown chunking.
    
    Args:
        file_path: The absolute path to the file.
        query: Keywords to extract relevant sections.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        if not query:
            return content[:2000] + "\n...(Summary mode not implemented for RAG, please provide query)..."

        rag = get_rag()
        if not rag:
            return "Error: RAG dependencies not installed. Please install 'requests' and 'langchain-text-splitters'."

        # 1. 智能分块 (Mature Chunking)
        chunks = rag.split_markdown(content)
        if not chunks:
            return content[:1000] # Fallback

        # 2. 向量化 (Embedding)
        # 调用云端 API 批量获取 embeddings
        try:
            doc_vecs = rag.embed_texts(chunks) # np.ndarray
            query_vec = rag.embed_texts([query])[0] # np.ndarray
        except Exception as e:
             return f"Error calling Embedding API: {str(e)}"
        
        # 3. 相似度计算 (Cosine Similarity)
        scores = rag.compute_similarity(query_vec, doc_vecs)
        
        # 4. 获取 Top 3
        top_k_indices = np.argsort(scores)[::-1][:3]
        
        output = [f"Found relevant sections for '{query}' (Semantic Search):\n"]
        for idx in top_k_indices:
            score = scores[idx]
            if score < 0.3: # 相似度阈值过滤
                continue
            output.append(f"--- Section (Similarity: {score:.2f}) ---")
            output.append(chunks[idx])
            output.append("\n")
            
        if len(output) == 1:
            return f"No highly relevant sections found for '{query}' (Low similarity)."
            
        return "\n".join(output)

    except Exception as e:
        import traceback
        return f"Error processing file: {str(e)}\n{traceback.format_exc()}"

if __name__ == "__main__":
    # 这种方式允许直接运行脚本启动 Server
    mcp.run()

def main():
    """Entry point for uv/pip scripts"""
    mcp.run()
