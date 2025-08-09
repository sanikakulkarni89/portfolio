import json
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any

# Your portfolio data (paste your JSON data here)
portfolio_data = [
        {
            "type": "project",
            "title": "µSocial - Event-Driven Microservices Architecture for Distributed Social Networks",
            "category": "distributed_systems",
            "content": "Production-grade microservices ecosystem implementing a scalable social media platform using modern distributed systems patterns. Features event-driven architecture with Apache Kafka, CQRS implementation, polyglot persistence (PostgreSQL, MongoDB, Neo4j, Redis), saga pattern for distributed transactions, event sourcing, API Gateway with OAuth 2.0/JWT, and service mesh with Istio. Includes circuit breaker patterns, eventual consistency, distributed caching, and containerized deployment with Kubernetes."
        },
        {
            "type": "project",
            "title": "Stock Trading and Portfolio Management App",
            "category": "web_development",
            "content": "Built a responsive full-stack application using ReactJS, Spring Boot and MongoDB for portfolio management. Introduced key features such as stock search, watchlist manager, buying and selling capabilities, and portfolio tracking."
        },
        {
            "type": "project",
            "title": "Event Booking and Management App",
            "category": "web_development",
            "content": "Designed a responsive full-stack application using Angular, NodeJS, MongoDB with searching and booking features. Implemented key functionalities such as event searching, venue display, and comprehensive booking management. Successfully deployed on Google Cloud Platform for reliable and scalable performance."
        },
        {
            "type": "project",
            "title": "Flight Delay Prediction",
            "category": "machine_learning",
            "content": "Developed a machine learning model to predict flight delays up to four days in advance with an accuracy of 65%. Analyzed historical flight data and weather patterns and applied feature engineering techniques such as data cleaning, transformation, and reduction, employing models like Random Forest and Gradient Boosting to increase prediction accuracy."
        },
        {
            "type": "project",
            "title": "Multiprogramming Operating System using C++",
            "category": "systems_programming",
            "content": "Simulated an OS by programming operations including data read/write, moving data in and out of registers, etc. Enabled execution of multiple tasks, input spooling, output spooling, paging, and error handling, with an efficiency boost of up to 30%."
        },
        {
            "type": "project",
            "title": "Reselling Used Textbooks App",
            "category": "web_development",
            "content": "Implemented a RESTful API in Java to handle a used textbook inventory, applying design patterns and core concepts. Optimized book resale operations by 30% through clean architecture and modular design, supporting dynamic pricing with a 10% depreciation model."
        },
        {
            "type": "project",
            "title": "Medical Chatbot for Symptom Analysis",
            "category": "artificial_intelligence",
            "content": "Designed a chatbot using advanced NLP techniques and Large Language Models (LLMs) such as GPT-4 and BERT, incorporating text preprocessing (tokenization, normalization, stemming) and models like Naive Bayes, RNNs, and transformers. Implemented functionalities for accurate symptom checking, achieving 85% accuracy in evaluations, and real-time medical report summarization."
        },
        {
            "type": "project",
            "title": "Generation M",
            "category": "artificial_intelligence",
            "content": "Engineered a web application by making use of advanced models, including Imagen, Tacotron 2, DALL-E, and Stable Diffusion, to efficiently generate images and audio from text, addressing growing demand for rapid media production with an accuracy of 85%."
        },
        {
            "type": "project",
            "title": "Advanced Game Engine Optimization: Physics Simulation and Frustum Culling in C++",
            "category": "game_development",
            "content": "Implemented frustum culling with AABB, enhancing framerate in 3D rendering scenarios with 100+ characters. Designed a multi-threaded Physics-Component system, integrating collision, gravity for improved 3D gameplay realism."
        },
        {
            "type": "project",
            "title": "Plagiarism Detector",
            "category": "machine_learning",
            "content": "Engineered a robust plagiarism detection system leveraging advanced TF-IDF and vectorization methodologies, employing sophisticated feature extraction techniques on textual data to quantify and analyze plagiarism percentages within documents."
        },
        {
            "type": "project",
            "title": "ReadEasy",
            "category": "web_development",
            "content": "Designed and constructed an online web application facilitating the purchase, sale, and rental of educational books, leveraging the XAMPP framework to create a dynamic and resourceful platform for educational materials."
        },
        {
            "type": "project",
            "title": "Bot Detection System in Twitter",
            "category": "machine_learning",
            "content": "Built a system to distinguish between real users and bot accounts on Twitter through systematic analysis of machine learning algorithms. Performed dataset extraction, preprocessing, training and evaluation of ML models to develop a high-accuracy detection algorithm with real-world applicability for social media mining and cybersecurity."
        },
        {
            "type": "project",
            "title": "Headliner - Personalized News App",
            "category": "web_development",
            "content": "Built using React.js with TailwindCSS for responsive design and Node.js for server-side functionality. The news app pulls content from multiple sources using News API integration, implements user preference settings, and features a clean, intuitive interface designed for quick news consumption."
        },
        {
            "type": "project",
            "title": "TaskCLI: Productivity CLI Tool",
            "category": "software_development",
            "content": "Developed a command-line task management tool using Python with Click library for CLI interface and SQLite for data persistence. Features include adding tasks with priorities and due dates, flexible filtering and listing, task completion tracking, deletion and management, and productivity statistics visualization."
        },
        {
            "type": "project",
            "title": "LAMP - Job Search Efficiency App",
            "category": "mobile_development",
            "content": "Created a React Native mobile app implementing Steve Dalton's LAMP method for efficient job searching. Built with TypeScript, SVG graphics, and Animated API for smooth transitions. Features guided timers for List, Alumni, Motivation, and Postings phases, progress tracking with visual indicators, calendar integration, and completion statistics."
        },
        {
            "type": "project",
            "title": "LeetCode Problem Recommender",
            "category": "machine_learning",
            "content": "Built an intelligent recommender system using Python, Pandas, and Scikit-Learn to suggest LeetCode problems based on skill level and learning goals. Uses cosine similarity for topic matching and includes personalized recommendations, adaptive difficulty progression, smart algorithm with variety control, and progress visualization with Matplotlib."
        },
        {
            "type": "project",
            "title": "Real-Time Object Detection",
            "category": "computer_vision",
            "content": "Implemented YOLO (You Only Look Once) algorithm for real-time object detection using OpenCV, Python, and YOLOv3 pre-trained model. Detects 80+ object classes, draws bounding boxes with confidence scores, applies non-maximum suppression, and works with various image formats and resolutions."
        },
        {
            "type": "project",
            "title": "ECS Research Day Automation Challenge",
            "category": "web_development",
            "content": "Comprehensive system for automating college research poster judging process. Built intelligent judge assignment using web scraping and cosine similarity matching, developed Angular mobile scoring interface with Supabase database integration, and implemented fair ranking algorithm with bias correction through score normalization. Complete end-to-end workflow from assignment through scoring to final ranking."
        },
        {
            "type": "skill",
            "title": "Programming Languages",
            "category": "technical_skills",
            "content": "Python, Java, C, C++, JavaScript, HTML, CSS, Dart, SQL"
        },
        {
            "type": "skill",
            "title": "Database Technologies",
            "category": "technical_skills",
            "content": "MySQL, Azure Cloud, PostgreSQL, Amazon Redshift, Oracle, MongoDB"
        },
        {
            "type": "skill",
            "title": "Platform Experience",
            "category": "technical_skills",
            "content": "Android Studio, UNIX, Google Cloud Platform, AWS, Firebase, MongoDB, Supabase"
        },
        {
            "type": "skill",
            "title": "Frameworks and Libraries",
            "category": "technical_skills",
            "content": "Flutter, Angular, React.JS, Node.JS, Vue.js, Spring Boot, Flask, Git, TensorFlow, Scikit-Learn, Hibernate"
        },
        {
            "type": "skill",
            "title": "Core Computer Science Concepts",
            "category": "technical_skills",
            "content": "Design and Analysis of Algorithms, Object Oriented Programming, Database Management Systems, Computer Architecture, Machine Learning, Object Oriented Design, Natural Language Processing"
        },
        {
            "type": "skill",
            "title": "Soft Skills",
            "category": "soft_skills",
            "content": "Problem Solving, Teamwork, Communication, Adaptability, Time Management, Leadership"
        },
        {
            "type": "education",
            "title": "Diploma in Information Technology",
            "category": "education",
            "content": "Completed a Diploma in Information Technology from PCP, Pune. Focused on core IT subjects including programming, database management, and web development. Engaged in practical projects to apply theoretical knowledge in real-world scenarios, enhancing skills in software development and system design."
        },
        {
            "type": "education",
            "title": "Bachelors of Technology in Computer Science",
            "category": "education",
            "content": "Completed a B.Tech in Computer Science from DYPIU with a minor in Artificial Intelligence and Machine Learning and a focus on software development, machine learning, and distributed systems. Engaged in various projects and internships to apply theoretical knowledge in practical scenarios, enhancing skills in programming, system design, and problem-solving."
        },
        {
            "type": "education",
            "title": "Masters of Science in Computer Science",
            "category": "education",
            "content": "Completed Masters in Computer Science from Syracuse University. Focused on advanced topics such as deep learning, natural language processing, and distributed systems. Engaged in research projects and internships to deepen understanding of AI applications and enhance programming and system design skills."
        },
        {
            "type": "experience",
            "title": "Software Development Intern at Wajooba",
            "category": "work_experience",
            "content": "Engineered a comprehensive payment module, integrating dual payment systems across two distinct Flutter applications, reducing checkout abandonment by 23% and increasing transaction completion rate by 17% within 3 months of implementation. Partnered with clients to refine UI features through 12 feedback cycles, leading to a 28% increase in user satisfaction scores and a 4.2 to 4.7 star rating improvement across both platforms."
        },
        {
            "type": "experience",
            "title": "Software Development Intern at KP Marketing Agency",
            "category": "work_experience",
            "content": "Developed and executed 8 responsive websites using WordPress and front-end technologies, improving user experience across devices and reducing bounce rates by 31% while increasing average session duration by 2.4 minutes. Managed data architecture with MySQL and MongoDB, optimizing database queries that improved page load times by 42% and ensured robust web functionality through implementation of automated testing protocols that reduced post-deployment bugs by 87%."
        },
        {
            "type": "experience",
            "title": "Software Development Intern at ACP IT Zone",
            "category": "work_experience",
            "content": "Contributed to the development of a real-time coding platform, enhancing user engagement by 35% through the implementation of interactive coding challenges and live coding sessions. Collaborated with a team of developers to optimize backend performance, resulting in a 50% reduction in server response time and improved overall system reliability."
        
        },
        {
            "type": "experience",
            "title": "Research Assistant at Syracuse University",
            "category": "work_experience",
            "content": "Developed a comprehensive online coding platform, enhancing user engagement by 40% through the implementation of interactive coding challenges and live coding sessions. Collaborated with a team of developers to optimize backend performance, resulting in a 50% reduction in server response time and improved overall system reliability." 
        },
        {
            "type": "course",
            "title": "Introduction to Design and Analysis of Algorithms",
            "category": "bachelors",
            "content": "Core computer science course covering algorithmic design techniques, complexity analysis, and optimization strategies"
        },
        {
            "type": "course",
            "title": "Digital Communications",
            "category": "bachelors",
            "content": "Study of digital signal processing, modulation techniques, and communication system design"
        },
        {
            "type": "course",
            "title": "Computer Organization",
            "category": "bachelors",
            "content": "Computer hardware architecture, processor design, memory systems, and instruction execution"
        },
        {
            "type": "course",
            "title": "Computer Networks",
            "category": "bachelors",
            "content": "Network protocols, TCP/IP, network security, distributed systems, and internet architecture"
        },
        {
            "type": "course",
            "title": "Technology Management and Commercialization",
            "category": "bachelors",
            "content": "Business aspects of technology development, product management, and technology transfer"
        },
        {
            "type": "course",
            "title": "Design Project",
            "category": "bachelors",
            "content": "Hands-on project course applying engineering design principles to solve real-world problems"
        },
        {
            "type": "course",
            "title": "Introduction to Intelligent Systems",
            "category": "bachelors",
            "content": "Fundamentals of artificial intelligence, expert systems, and knowledge representation"
        },
        {
            "type": "course",
            "title": "Database Systems",
            "category": "bachelors",
            "content": "Database design, SQL, relational algebra, normalization, and database management systems"
        },
        {
            "type": "course",
            "title": "Systems Software",
            "category": "bachelors",
            "content": "Operating systems, compilers, assemblers, and system-level programming"
        },
        {
            "type": "course",
            "title": "Embedded Systems Development",
            "category": "bachelors",
            "content": "Microcontroller programming, real-time systems, and embedded software development"
        },
        {
            "type": "course",
            "title": "Design Project-II",
            "category": "bachelors",
            "content": "Advanced project course building on design principles with more complex engineering challenges"
        },
        {
            "type": "course",
            "title": "Systems Security",
            "category": "bachelors",
            "content": "Computer security, cryptography, network security, and cybersecurity principles"
        },
        {
            "type": "course",
            "title": "Software Engineering and Project Management",
            "category": "bachelors",
            "content": "Software development lifecycle, project management methodologies, and team collaboration"
        },
        {
            "type": "course",
            "title": "DSP Systems",
            "category": "bachelors",
            "content": "Digital signal processing algorithms, filter design, and signal analysis techniques"
        },
        {
            "type": "course",
            "title": "Fundamentals of AI/ML",
            "category": "bachelors",
            "content": "Introduction to artificial intelligence and machine learning concepts and applications"
        },
        {
            "type": "course",
            "title": "Principle of Data Science & Engineering",
            "category": "bachelors",
            "content": "Data analysis, statistical methods, data mining, and big data processing techniques"
        },
        {
            "type": "course",
            "title": "Human Physiology",
            "category": "bachelors",
            "content": "Study of human biological systems and physiological processes"
        },
        {
            "type": "course",
            "title": "Deep Neural Network",
            "category": "bachelors",
            "content": "Advanced neural network architectures, deep learning algorithms, and model training"
        },
        {
            "type": "course",
            "title": "Game Theory",
            "category": "bachelors",
            "content": "Mathematical analysis of strategic decision-making and competitive scenarios"
        },
        {
            "type": "course",
            "title": "High Performance Computing",
            "category": "bachelors",
            "content": "Parallel computing, distributed systems, and optimization for computational performance"
        },
        {
            "type": "course",
            "title": "Quantum Computing",
            "category": "bachelors",
            "content": "Quantum mechanics principles applied to computing, quantum algorithms, and quantum systems"
        },
        {
            "type": "course",
            "title": "Marketing Models",
            "category": "bachelors",
            "content": "Mathematical and statistical models for marketing analysis and consumer behavior"
        },
        {
            "type": "course",
            "title": "Research Internship",
            "category": "bachelors",
            "content": "Hands-on research experience in academic or industry setting"
        },
        {
            "type": "course",
            "title": "AI Processors & Architecture",
            "category": "bachelors",
            "content": "Specialized hardware for artificial intelligence applications and neural network processing"
        },
        {
            "type": "course",
            "title": "AI Applications and Ethics",
            "category": "bachelors",
            "content": "Practical applications of AI technology and ethical considerations in AI development"
        },
        {
            "type": "course",
            "title": "Blockchain",
            "category": "bachelors",
            "content": "Distributed ledger technology, cryptocurrency, smart contracts, and blockchain applications"
        },
        {
            "type": "course",
            "title": "UI/UX Design",
            "category": "bachelors",
            "content": "User interface design principles, user experience research, and human-computer interaction"
        },
        {
            "type": "course",
            "title": "Principles of Social Media and Data Mining",
            "category": "masters",
            "content": "Analysis of social networks, data mining techniques, and social media analytics"
        },
        {
            "type": "course",
            "title": "Principles of Operating Systems",
            "category": "masters",
            "content": "Advanced operating system concepts, process management, memory management, and file systems"
        },
        {
            "type": "course",
            "title": "Computer Architecture",
            "category": "masters",
            "content": "Advanced computer system design, processor architecture, and performance optimization"
        },
        {
            "type": "course",
            "title": "Applied Natural Language Processing",
            "category": "masters",
            "content": "Text processing, language models, sentiment analysis, and NLP applications"
        },
        {
            "type": "course",
            "title": "Object Oriented Design",
            "category": "masters",
            "content": "Advanced object-oriented programming principles, design patterns, and software architecture"
        },
        {
            "type": "course",
            "title": "Database Management Systems",
            "category": "masters",
            "content": "Advanced database concepts, query optimization, distributed databases, and database theory"
        },
        {
            "type": "course",
            "title": "Machine Learning and Algorithms",
            "category": "masters",
            "content": "Advanced machine learning techniques, algorithmic approaches, and model optimization"
        },
        {
            "type": "course",
            "title": "Design and Analysis of Algorithms",
            "category": "masters",
            "content": "Advanced algorithmic design, complexity theory, and optimization techniques"
        },
        {
            "type": "course",
            "title": "Introduction to Machine Learning and Algorithms",
            "category": "masters",
            "content": "Foundational machine learning concepts, algorithms, and practical applications"
        }
 ]




class PortfolioRAGProcessor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the RAG processor with embedding model and vector DB"""
        self.embedding_model = SentenceTransformer(model_name)
        self.client = chromadb.Client()
        self.collection = None
        
        
    def create_enhanced_chunks(self, data: List[Dict]) -> List[Dict]:
        """Create intelligently chunked data for better retrieval"""
        chunks = []
        
        for item in data:
            item_type = item.get('type', 'unknown')
            title = item.get('title', 'Untitled')
            category = item.get('category', 'general')
            content = item.get('content', '')
            
            # Create different chunk strategies based on type
            if item_type == 'project':
                chunks.extend(self._chunk_project(item))
            elif item_type == 'skill':
                chunks.extend(self._chunk_skill(item))
            elif item_type == 'experience':
                chunks.extend(self._chunk_experience(item))
            elif item_type == 'education':
                chunks.extend(self._chunk_education(item))
            elif item_type == 'course':
                chunks.extend(self._chunk_course(item))
                
        return chunks
    
    def _chunk_project(self, project: Dict) -> List[Dict]:
        """Create multiple chunks for projects to improve retrieval"""
        chunks = []
        title = project['title']
        category = project['category']
        content = project['content']
        
        # Main project chunk
        main_chunk = {
            'id': f"project_{self._generate_id(title)}",
            'text': f"Project: {title}\nCategory: {category}\nDescription: {content}",
            'metadata': {
                'type': 'project',
                'title': title,
                'category': category,
                'chunk_type': 'main'
            }
        }
        chunks.append(main_chunk)
        
        # Technology-focused chunk if technologies are mentioned
        tech_keywords = self._extract_technologies(content)
        if tech_keywords:
            tech_chunk = {
                'id': f"project_tech_{self._generate_id(title)}",
                'text': f"Project {title} uses technologies: {', '.join(tech_keywords)}. {content}",
                'metadata': {
                    'type': 'project_technology',
                    'title': title,
                    'category': category,
                    'technologies': ', '.join(tech_keywords),  # Convert list to string
                    'chunk_type': 'technology'
                }
            }
            chunks.append(tech_chunk)
            
        # Achievement-focused chunk if metrics are mentioned
        if any(char.isdigit() and '%' in content for char in content):
            achievement_chunk = {
                'id': f"project_achievement_{self._generate_id(title)}",
                'text': f"Project achievements for {title}: {content}",
                'metadata': {
                    'type': 'project_achievement',
                    'title': title,
                    'category': category,
                    'chunk_type': 'achievement'
                }
            }
            chunks.append(achievement_chunk)
            
        return chunks
    
    def _chunk_skill(self, skill: Dict) -> List[Dict]:
        """Create skill chunks with individual technology mentions"""
        chunks = []
        title = skill['title']
        category = skill['category']
        content = skill['content']
        
        # Main skill chunk
        main_chunk = {
            'id': f"skill_{self._generate_id(title)}",
            'text': f"Skill Category: {title}\nSkills: {content}",
            'metadata': {
                'type': 'skill',
                'category': category,
                'skill_category': title,
                'chunk_type': 'main'
            }
        }
        chunks.append(main_chunk)
        
        # Individual skill chunks for better matching
        individual_skills = [s.strip() for s in content.split(',')]
        for i, skill_item in enumerate(individual_skills):  # Fixed: use enumerate()
            if skill_item:
                # Added uuid to ensure uniqueness
                unique_id = str(uuid.uuid4())[:8]
                individual_chunk = {
                    'id': f"individual_skill_{self._generate_id(skill_item)}_{unique_id}",
                    'text': f"I have experience with {skill_item} in {title.lower()}",
                    'metadata': {
                        'type': 'individual_skill',
                        'skill': skill_item,
                        'category': category,
                        'skill_category': title,
                        'chunk_type': 'individual'
                    }
                }
                chunks.append(individual_chunk)
            
        return chunks
    
    def _chunk_experience(self, experience: Dict) -> List[Dict]:
        """Create experience chunks with role and achievement focus"""
        chunks = []
        title = experience['title']
        category = experience['category']
        content = experience['content']
        
        # Main experience chunk
        main_chunk = {
            'id': f"experience_{self._generate_id(title)}",
            'text': f"Work Experience: {title}\n{content}",
            'metadata': {
                'type': 'experience',
                'role': title,
                'category': category,
                'chunk_type': 'main'
            }
        }
        chunks.append(main_chunk)
        
        # Achievement-focused chunk
        achievement_chunk = {
            'id': f"experience_achievement_{self._generate_id(title)}",
            'text': f"Achievements in role {title}: {content}",
            'metadata': {
                'type': 'experience_achievement',
                'role': title,
                'category': category,
                'chunk_type': 'achievement'
            }
        }
        chunks.append(achievement_chunk)
        
        return chunks
    
    def _chunk_education(self, education: Dict) -> List[Dict]:
        """Create education chunks"""
        chunks = []
        title = education['title']
        category = education['category']
        content = education['content']
        
        chunk = {
            'id': f"education_{self._generate_id(title)}",
            'text': f"Education: {title}\n{content}",
            'metadata': {
                'type': 'education',
                'degree': title,
                'category': category,
                'chunk_type': 'main'
            }
        }
        chunks.append(chunk)
        
        return chunks
    
    def _chunk_course(self, course: Dict) -> List[Dict]:
        """Create course chunks grouped by degree level"""
        chunks = []
        title = course['title']
        category = course['category']
        content = course['content']
        
        chunk = {
            'id': f"course_{self._generate_id(title)}",
            'text': f"Course: {title} ({category} level)\nDescription: {content}",
            'metadata': {
                'type': 'course',
                'course_name': title,
                'degree_level': category,
                'chunk_type': 'main'
            }
        }
        chunks.append(chunk)
        
        return chunks
    
    def _extract_technologies(self, content: str) -> List[str]:
        """Extract technology keywords from content"""
        tech_patterns = [
            'React', 'Angular', 'Vue.js', 'Node.js', 'Python', 'Java', 'JavaScript',
            'TypeScript', 'Spring Boot', 'Flask', 'Django', 'MongoDB', 'PostgreSQL',
            'MySQL', 'Redis', 'Docker', 'Kubernetes', 'AWS', 'GCP', 'Azure',
            'TensorFlow', 'PyTorch', 'Scikit-Learn', 'OpenCV', 'BERT', 'GPT',
            'Apache Kafka', 'Microservices', 'REST', 'GraphQL', 'JWT', 'OAuth',
            'Git', 'Flutter', 'React Native', 'C++', 'C', 'SQL', 'NoSQL',
            'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
            'Blockchain', 'Unity', 'Game Development', 'YOLO', 'Stable Diffusion'
        ]
        
        found_techs = []
        content_lower = content.lower()
        for tech in tech_patterns:
            if tech.lower() in content_lower:
                found_techs.append(tech)
        
        return found_techs
    
    def _generate_id(self, text: str) -> str:
        """Generate a clean ID from text"""
        return text.lower().replace(' ', '_').replace('-', '_').replace('/', '_')[:50]
    
    def create_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Create embeddings for all chunks"""
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
            
        return chunks
    
    def setup_vector_database(self, collection_name: str = "portfolio_rag"):
        """Initialize ChromaDB collection"""
        try:
            # Delete existing collection if it exists
            self.client.delete_collection(collection_name)
        except:
            pass
            
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "Portfolio RAG system for interactive Q&A"}
        )
        print(f"Created collection: {collection_name}")
    
    def store_chunks(self, chunks: List[Dict]):
        """Store chunks in vector database"""
        if not self.collection:
            raise ValueError("Vector database not initialized. Call setup_vector_database() first.")
        
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]
        
        # Store in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
        
        print(f"Stored {len(chunks)} chunks in vector database")
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search the vector database"""
        if not self.collection:
            raise ValueError("Vector database not initialized.")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0] if results['distances'] else []
        }
    
    def process_portfolio(self, data: List[Dict], collection_name: str = "portfolio_rag"):
        """Complete pipeline: chunk, embed, and store portfolio data"""
        print("Starting portfolio processing pipeline...")
        
        # Step 1: Create chunks
        chunks = self.create_enhanced_chunks(data)
        print(f"Created {len(chunks)} chunks")
        
        # Step 2: Create embeddings
        chunks_with_embeddings = self.create_embeddings(chunks)
        
        # Step 3: Setup vector database
        self.setup_vector_database(collection_name)
        
        # Step 4: Store chunks
        self.store_chunks(chunks_with_embeddings)
        
        print("Portfolio processing complete!")
        return chunks_with_embeddings

# Example usage and testing
def main():
    # Initialize processor
    processor = PortfolioRAGProcessor()
    
    # Process your portfolio data
    # Make sure to replace 'portfolio_data' with your actual JSON data
    chunks = processor.process_portfolio(portfolio_data)
    
    # Test the search functionality
    print("\n" + "="*50)
    print("TESTING SEARCH FUNCTIONALITY")
    print("="*50)
    
    test_queries = [
        "What experience do you have with React?",
        "Tell me about your machine learning projects",
        "What databases have you worked with?",
        "Do you have experience with microservices?",
        "What's your educational background?",
        "Have you worked with any AI models?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        results = processor.search(query, n_results=3)
        
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
            print(f"{i+1}. [{meta['type']}] {doc[:100]}...")
            if len(results['distances']) > i:
                print(f"   Similarity: {1 - results['distances'][i]:.3f}")
    
    print(f"\nTotal chunks created: {len(chunks)}")
    print("RAG system is ready for integration with Llama!")

if __name__ == "__main__":
    main()