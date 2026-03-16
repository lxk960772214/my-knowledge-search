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
parser.add_argument('--exclude-dirs', type=str, default='',
                    help='Comma-separated list of directories to exclude')
parser.add_argument('--exclude-files', type=str, default='',
                    help='Comma-separated list of files to exclude')

# 初始化 MCP Server
mcp = FastMCP("LocalKnowledgeSearch")

# 获取搜索目录和过滤配置
SEARCH_ROOTS = []
GLOBAL_EXCLUDE_DIRS = set()
GLOBAL_EXCLUDE_FILES = set()

def _init_config():
    global SEARCH_ROOTS, GLOBAL_EXCLUDE_DIRS, GLOBAL_EXCLUDE_FILES
    
    # 使用 argparse 解析参数，但要注意 mcp.run() 可能会干扰参数解析
    # 这里我们手动处理 sys.argv 以避免与 FastMCP 的参数冲突
    # FastMCP 通常接管了 server 的运行，但我们可以提取我们需要的参数
    
    args = sys.argv[1:]
    clean_args = []
    
    # 简单的手动解析，提取 --exclude-dirs 和 --exclude-files
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--exclude-dirs':
            if i + 1 < len(args):
                GLOBAL_EXCLUDE_DIRS.update([d.strip() for d in args[i+1].split(',') if d.strip()])
                i += 2
            else:
                i += 1
        elif arg.startswith('--exclude-dirs='):
            val = arg.split('=', 1)[1]
            GLOBAL_EXCLUDE_DIRS.update([d.strip() for d in val.split(',') if d.strip()])
            i += 1
        elif arg == '--exclude-files':
            if i + 1 < len(args):
                GLOBAL_EXCLUDE_FILES.update([f.strip() for f in args[i+1].split(',') if f.strip()])
                i += 2
            else:
                i += 1
        elif arg.startswith('--exclude-files='):
            val = arg.split('=', 1)[1]
            GLOBAL_EXCLUDE_FILES.update([f.strip() for f in val.split(',') if f.strip()])
            i += 1
        else:
            if os.path.isdir(arg):
                SEARCH_ROOTS.append(os.path.abspath(arg))
            i += 1
            
    if not SEARCH_ROOTS:
         SEARCH_ROOTS = []
         
    print(f"Initialized search roots: {SEARCH_ROOTS}", file=sys.stderr)
    print(f"Global exclude dirs: {GLOBAL_EXCLUDE_DIRS}", file=sys.stderr)
    print(f"Global exclude files: {GLOBAL_EXCLUDE_FILES}", file=sys.stderr)

_init_config()

import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# --- Semantic RAG Core ---
class LocalRAG:
    def __init__(self):
        if not RAG_AVAILABLE:
            return
        # Cloud Embedding Configuration
        self.api_url = os.getenv("EMBEDDING_API_URL", "https://openrouter.ai/api/v1/embeddings")
        self.api_key = os.getenv("EMBEDDING_API_KEY")
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "baai/bge-m3")
        
        if not self.api_key:
            raise ValueError("EMBEDDING_API_KEY environment variable is required for RAG")
        
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
            
            # Check for HTTP errors and log full response if error
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f"HTTP Error: {e}", file=sys.stderr)
                print(f"Full Response Body: {response.text}", file=sys.stderr)
                raise e
            
            data = response.json()
            if 'data' not in data:
                 print(f"Embedding API Unexpected Response: {data}", file=sys.stderr)
                 raise ValueError(f"API response missing 'data' field. Response: {str(data)[:200]}...")
                 
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
def search_files(query: str, exclude_dirs: str = None, exclude_files: str = None) -> str:
    """
    Search for files using fuzzy keywords. 
    Matches filenames and YAML Front Matter metadata (tags, aliases, clients) with high priority.
    PERFORMANCE MODE: Skips full content search.
    
    Args:
        query: The search term (e.g., "python", "byd", "migration").
        exclude_dirs: Optional comma-separated list of directory names to exclude (e.g. "node_modules,test").
        exclude_files: Optional comma-separated list of filenames to exclude.
    """
    results = []
    query_lower = query.lower()
    
    # Initialize ignore sets with defaults and global config
    ignored_dirs_set = {'.git', 'node_modules', '__pycache__', '.idea', '.vscode', 'venv', 'env'}
    ignored_dirs_set.update(GLOBAL_EXCLUDE_DIRS)
    if exclude_dirs:
        ignored_dirs_set.update([d.strip() for d in exclude_dirs.split(',') if d.strip()])
        
    ignored_files_set = {'.DS_Store'}
    ignored_files_set.update(GLOBAL_EXCLUDE_FILES)
    if exclude_files:
        ignored_files_set.update([f.strip() for f in exclude_files.split(',') if f.strip()])
    
    print(f"Searching for '{query}' in {len(SEARCH_ROOTS)} directories (Performance Mode)...")
    if exclude_dirs:
        print(f"Ignoring directories: {ignored_dirs_set}")

    for root_dir in SEARCH_ROOTS:
        if not os.path.exists(root_dir):
            continue
            
        # Search in current directory
        for root, dirs, files in os.walk(root_dir):
            # Modify dirs in-place to prune search tree
            # 支持 1. 目录名在忽略列表中 2. 目录完整路径在忽略列表中
            i = 0
            while i < len(dirs):
                d = dirs[i]
                full_dir_path = os.path.join(root, d)
                if d.startswith('.') or d in ignored_dirs_set or full_dir_path in ignored_dirs_set:
                    del dirs[i]
                else:
                    i += 1
            
            current_dir_name = os.path.basename(root)
            dir_matches = query_lower in current_dir_name.lower()
            
            # Pre-calculate ancestor matches (Optimization)
            ancestor_score = 0
            ancestor_reason = ""
            
            if not dir_matches:
                parent_dir = os.path.dirname(root)
                grandparent_dir = os.path.dirname(parent_dir)
                
                parent_name = os.path.basename(parent_dir)
                grandparent_name = os.path.basename(grandparent_dir)
                
                if query_lower in parent_name.lower():
                    ancestor_score = 10
                    ancestor_reason = f"AncestorDirMatch({parent_name} - Readme)"
                elif query_lower in grandparent_name.lower():
                    ancestor_score = 10
                    ancestor_reason = f"AncestorDirMatch({grandparent_name} - Readme)"

            for file in files:
                file_path = os.path.join(root, file)
                
                # 排除系统文件、非MD文件以及指定忽略的文件/路径
                if (file.startswith('.') or 
                    not file.lower().endswith('.md') or 
                    file in ignored_files_set or 
                    file_path in ignored_files_set):
                    continue
                
                score = 0
                match_reasons = []
                is_readme = file.lower().startswith('readme')
                
                # 1. 目录名匹配
                if dir_matches:
                    if is_readme:
                        score += 30
                        match_reasons.append(f"DirectoryMatch({current_dir_name} - Readme)")
                elif is_readme and ancestor_score > 0:
                    score += ancestor_score
                    match_reasons.append(ancestor_reason)
                        
                # 2. 文件名匹配
                if query_lower in file.lower():
                    score += 30
                    match_reasons.append("Filename")
                
                # 3. 仅匹配 YAML Front Matter (文件头)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        head_content = f.read(500) 
                        if head_content.startswith('---'):
                            end_idx = head_content.find('---', 3)
                            if end_idx != -1:
                                yaml_block = head_content[3:end_idx].lower()
                                if query_lower in yaml_block:
                                    score += 10
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
             print(f"RAG Embedding failed: {e}. Falling back to keyword search.", file=sys.stderr)
             # Fallback: Simple keyword filtering if embedding fails
             fallback_output = [f"Warning: Semantic search failed ({str(e)}). Showing sections containing '{query}':\n"]
             found_any = False
             for chunk in chunks:
                 if query.lower() in chunk.lower():
                     fallback_output.append("--- Section (Keyword Match) ---")
                     fallback_output.append(chunk)
                     fallback_output.append("\n")
                     found_any = True
             
             if not found_any:
                  return f"Semantic search failed and no exact keyword matches found for '{query}'.\nHere is the beginning of the file:\n\n{content[:1000]}..."
             
             return "\n".join(fallback_output)
        
        # 3. 相似度计算 (Cosine Similarity)
        scores = rag.compute_similarity(query_vec, doc_vecs)
        
        # 4. 获取 Top 2
        top_k_indices = np.argsort(scores)[::-1][:2]
        
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
