import requests
import json
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from portfolio_rag import PortfolioRAGProcessor, portfolio_data

class PortfolioRAGChat:
    def __init__(self, 
                 collection_name: str = "portfolio_rag",
                 model_name: str = "llama3.2",
                 ollama_url: str = "http://localhost:11434"):
        """Initialize RAG chat system with Llama integration"""
        
        # Initialize vector database
        self.client = chromadb.Client()
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"✅ Connected to existing collection: {collection_name}")
        except:
            print(f"❌ Collection '{collection_name}' not found. Creating new collection...")
            # Initialize the processor and create the collection
            processor = PortfolioRAGProcessor()
            # TODO: Replace the following line with actual portfolio data loading logic
            # portfolio_data = processor.load_portfolio_data()  # Make sure this method exists or replace with your data
            processor.process_portfolio(portfolio_data, collection_name)
            self.collection = self.client.get_collection(collection_name)
            print(f"✅ Created and populated new collection: {collection_name}")
    
        # Ollama configuration
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test Ollama connection
        self._test_ollama_connection()
        
    def _test_ollama_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json={
                                       "model": self.model_name,
                                       "prompt": "Hello",
                                       "stream": False
                                   }, timeout=20)
            if response.status_code == 200:
                print(f"✅ Ollama connection successful with model: {self.model_name}")
            else:
                print(f"❌ Ollama error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to Ollama at {self.ollama_url}")
            print("Make sure Ollama is running: ollama serve")
            raise
    
    def retrieve_context(self, query: str, n_results: int = 5) -> tuple:
        """Retrieve relevant context from vector database"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format context with metadata for better understanding
        contexts = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            context_type = meta.get('type', 'unknown')
            if context_type == 'project':
                contexts.append(f"PROJECT: {doc}")
            elif context_type == 'skill' or context_type == 'individual_skill':
                contexts.append(f"SKILL: {doc}")
            elif context_type == 'experience':
                contexts.append(f"EXPERIENCE: {doc}")
            elif context_type == 'education':
                contexts.append(f"EDUCATION: {doc}")
            elif context_type == 'course':
                contexts.append(f"COURSE: {doc}")
            else:
                contexts.append(doc)
        
        return contexts, results['metadatas'][0]
    
    def create_rag_prompt(self, query: str, context: List[str]) -> str:
        """Create a well-structured prompt for RAG"""
        context_text = "\n\n".join(context)
        
        prompt = f"""You are a helpful assistant answering questions about a software developer's portfolio and background. Use ONLY the provided context to answer questions. Be specific, accurate, and professional.

CONTEXT INFORMATION:
{context_text}

IMPORTANT INSTRUCTIONS:
- Answer based ONLY on the provided context
- Be specific about projects, technologies, and achievements
- If the context doesn't contain enough information, say so
- Use a professional but conversational tone
- Include specific details like metrics, technologies used, and outcomes when available
- If asked about experience with specific technologies, mention the projects where they were used

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return f"Error generating response: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def chat(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        """Main RAG chat function"""
        
        # Step 1: Retrieve relevant context
        if verbose:
            print("🔍 Retrieving relevant context...")
        
        context, metadata = self.retrieve_context(query, n_results=5)
        
        # Step 2: Create RAG prompt
        prompt = self.create_rag_prompt(query, context)
        
        if verbose:
            print("📝 Generated prompt:")
            print("-" * 50)
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print("-" * 50)
        
        # Step 3: Generate response
        if verbose:
            print("🤖 Generating response with Llama...")
        
        response = self.generate_response(prompt)
        
        return {
            "query": query,
            "response": response,
            "context_used": context,
            "metadata": metadata,
            "prompt": prompt if verbose else None
        }
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n" + "="*60)
        print("🤖 PORTFOLIO RAG CHAT - Powered by Llama")
        print("="*60)
        print("Ask me anything about the portfolio!")
        print("Commands: 'quit' to exit, 'verbose' to toggle detailed output")
        print("-" * 60)
        
        verbose = False
        
        while True:
            try:
                query = input("\n💬 You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'verbose':
                    verbose = not verbose
                    print(f"🔧 Verbose mode: {'ON' if verbose else 'OFF'}")
                    continue
                
                if not query:
                    continue
                
                print("🤖 Assistant: ", end="", flush=True)
                
                result = self.chat(query, verbose=verbose)
                print(result['response'])
                
                if verbose:
                    print(f"\n📊 Context sources used: {len(result['context_used'])}")
                    for i, meta in enumerate(result['metadata']):
                        print(f"  {i+1}. {meta.get('type', 'unknown')} - {meta.get('title', 'N/A')}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

def run_test_queries():
    """Run some test queries to verify the system works"""
    chat_system = PortfolioRAGChat()
    
    test_queries = [
        "What programming languages do you know?",
        "Tell me about your machine learning projects",
        "Do you have experience with React?",
        "What internships have you done?",
        "Have you worked with databases?",
        "What's your educational background?"
    ]
    
    print("\n" + "="*60)
    print("🧪 RUNNING TEST QUERIES")
    print("="*60)
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        print("-" * 40)
        
        result = chat_system.chat(query)
        print(f"🤖 Response: {result['response']}")
        print()

def main():
    """Main function to run the RAG chat system"""
    try:
        # Option 1: Run test queries
        print("Choose an option:")
        print("1. Run test queries")
        print("2. Start interactive chat")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            run_test_queries()
        elif choice == "2":
            chat_system = PortfolioRAGChat()
            chat_system.interactive_chat()
        else:
            print("Invalid choice. Running test queries...")
            run_test_queries()
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you've run the chunking script first")
        print("2. Ensure Ollama is running: ollama serve")
        print("3. Check if the model is pulled: ollama pull llama3.2")

if __name__ == "__main__":
    main()