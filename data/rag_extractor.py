import requests
import json
import PyPDF2
import docx
from bs4 import BeautifulSoup
import re
from datetime import datetime
from typing import List, Dict, Any
import time
import os

class ComprehensiveRAGExtractor:
    def __init__(self):
        self.extracted_data = []
        
    def extract_resume_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF resume"""
        chunks = []
        try:
            pdf_path = 'data/Sanika Resume August 2025.pdf'
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # Clean and chunk the resume text
                chunks = self._chunk_resume_text(text)
                
        except Exception as e:
            print(f"Error reading PDF: {e}")
            
        return chunks
    
    def extract_resume_docx(self, docx_path: str) -> List[Dict]:
        """Extract text from DOCX resume"""
        chunks = []
        try:
            doc = docx.Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            chunks = self._chunk_resume_text(text)
            
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            
        return chunks
    
    def _chunk_resume_text(self, text: str) -> List[Dict]:
        """Intelligently chunk resume text by sections"""
        chunks = []
        
        # Common resume sections
        sections = {
            'summary': r'(summary|objective|profile|about)',
            'experience': r'(experience|work|employment|professional)',
            'education': r'(education|academic|degree|university|college)',
            'skills': r'(skills|technical|technologies|expertise)',
            'projects': r'(projects|portfolio)',
            'achievements': r'(achievements|accomplishments|awards|honors)',
            'certifications': r'(certifications|certificates|licenses)'
        }
        
        text_lower = text.lower()
        
        # Try to identify sections
        for section_name, pattern in sections.items():
            section_matches = re.finditer(pattern, text_lower)
            for match in section_matches:
                # Extract text around the match (rough section extraction)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 500)
                section_text = text[start:end].strip()
                
                chunk = {
                    'type': 'resume',
                    'title': f'Resume - {section_name.title()}',
                    'category': 'resume',
                    'content': section_text,
                    'source': 'resume_document',
                    'section': section_name
                }
                chunks.append(chunk)
        
        # If no sections found, create general resume chunk
        if not chunks:
            chunk = {
                'type': 'resume',
                'title': 'Resume - General',
                'category': 'resume',
                'content': text.strip(),
                'source': 'resume_document',
                'section': 'general'
            }
            chunks.append(chunk)
        
        return chunks
    
    def extract_github_profile(self, github_username: str) -> List[Dict]:
        """Extract GitHub profile and repository data"""
        chunks = []
        base_url = "https://api.github.com"
        
        try:
            # Get user profile
            user_response = requests.get(f"{base_url}/users/{github_username}")
            if user_response.status_code == 200:
                user_data = user_response.json()
                
                # Profile chunk
                profile_content = f"""
                GitHub Profile: {user_data.get('name', github_username)}
                Bio: {user_data.get('bio', 'No bio available')}
                Location: {user_data.get('location', 'Not specified')}
                Public Repositories: {user_data.get('public_repos', 0)}
                Followers: {user_data.get('followers', 0)}
                Following: {user_data.get('following', 0)}
                Account Created: {user_data.get('created_at', 'Unknown')}
                Company: {user_data.get('company', 'Not specified')}
                """
                
                chunk = {
                    'type': 'github_profile',
                    'title': 'GitHub Profile Overview',
                    'category': 'github',
                    'content': profile_content.strip(),
                    'source': 'github_api'
                }
                chunks.append(chunk)
            
            # Get repositories
            repos_response = requests.get(f"{base_url}/users/{github_username}/repos?sort=updated&per_page=20")
            if repos_response.status_code == 200:
                repos = repos_response.json()
                
                for repo in repos:
                    if not repo.get('fork', False):  # Skip forked repos
                        repo_content = f"""
                        Repository: {repo['name']}
                        Description: {repo.get('description', 'No description')}
                        Language: {repo.get('language', 'Not specified')}
                        Stars: {repo.get('stargazers_count', 0)}
                        Forks: {repo.get('forks_count', 0)}
                        Last Updated: {repo.get('updated_at', 'Unknown')}
                        Topics: {', '.join(repo.get('topics', []))}
                        """
                        
                        chunk = {
                            'type': 'github_repository',
                            'title': f'GitHub Repository - {repo["name"]}',
                            'category': 'github',
                            'content': repo_content.strip(),
                            'source': 'github_api',
                            'repo_name': repo['name'],
                            'language': repo.get('language', ''),
                            'stars': repo.get('stargazers_count', 0)
                        }
                        chunks.append(chunk)
            
        except Exception as e:
            print(f"Error extracting GitHub data: {e}")
        
        return chunks
    
    def extract_leetcode_profile(self, leetcode_username: str) -> List[Dict]:
        """Extract LeetCode profile data"""
        chunks = []
        
        try:
            # LeetCode GraphQL API (unofficial)
            query = """
            query getUserProfile($username: String!) {
                matchedUser(username: $username) {
                    username
                    profile {
                        realName
                        aboutMe
                        countryName
                        company
                        ranking
                    }
                    submitStats {
                        acSubmissionNum {
                            difficulty
                            count
                            submissions
                        }
                        totalSubmissionNum {
                            difficulty
                            count
                            submissions
                        }
                    }
                }
            }
            """
            
            response = requests.post(
                'https://leetcode.com/graphql',
                json={
                    'query': query,
                    'variables': {'username': leetcode_username}
                },
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                user_data = data['data']['matchedUser']
                
                if user_data:
                    profile = user_data.get('profile', {})
                    stats = user_data.get('submitStats', {})
                    
                    # Profile information
                    profile_content = f"""
                    LeetCode Profile: {leetcode_username}
                    Real Name: {profile.get('realName', 'Not provided')}
                    About: {profile.get('aboutMe', 'No description')}
                    Country: {profile.get('countryName', 'Not specified')}
                    Company: {profile.get('company', 'Not specified')}
                    Ranking: {profile.get('ranking', 'Not available')}
                    """
                    
                    chunk = {
                        'type': 'leetcode_profile',
                        'title': 'LeetCode Profile',
                        'category': 'leetcode',
                        'content': profile_content.strip(),
                        'source': 'leetcode_api'
                    }
                    chunks.append(chunk)
                    
                    # Submission statistics
                    if stats and 'acSubmissionNum' in stats:
                        stats_content = "LeetCode Submission Statistics:\n"
                        for stat in stats['acSubmissionNum']:
                            difficulty = stat['difficulty']
                            count = stat['count']
                            stats_content += f"{difficulty}: {count} problems solved\n"
                        
                        chunk = {
                            'type': 'leetcode_stats',
                            'title': 'LeetCode Statistics',
                            'category': 'leetcode',
                            'content': stats_content.strip(),
                            'source': 'leetcode_api'
                        }
                        chunks.append(chunk)
                
        except Exception as e:
            print(f"Error extracting LeetCode data: {e}")
            print("Note: LeetCode API can be unreliable. Consider manual data entry.")
        
        return chunks
    
    def extract_linkedin_manual(self, linkedin_data_file: str) -> List[Dict]:
        """Process manually exported LinkedIn data"""
        chunks = []
        
        try:
            if linkedin_data_file.endswith('.json'):
                with open(linkedin_data_file, 'r') as file:
                    data = json.load(file)
                    
                    # Process different sections of LinkedIn export
                    sections = {
                        'profile': 'Basic profile information',
                        'positions': 'Work experience',
                        'education': 'Educational background',
                        'skills': 'Skills and endorsements',
                        'recommendations': 'Recommendations received'
                    }
                    
                    for section, description in sections.items():
                        if section in data:
                            chunk = {
                                'type': f'linkedin_{section}',
                                'title': f'LinkedIn - {description}',
                                'category': 'linkedin',
                                'content': json.dumps(data[section], indent=2),
                                'source': 'linkedin_export'
                            }
                            chunks.append(chunk)
            
        except Exception as e:
            print(f"Error processing LinkedIn data: {e}")
        
        return chunks
    
    def create_linkedin_template(self) -> Dict:
        """Create a template for manual LinkedIn data entry"""
        template = {
            "profile": {
                "name": "Your Full Name",
                "headline": "Your Professional Headline",
                "summary": "Your LinkedIn summary/about section",
                "location": "Your Location",
                "industry": "Your Industry"
            },
            "experience": [
                {
                    "title": "Job Title",
                    "company": "Company Name",
                    "duration": "Start Date - End Date",
                    "description": "Job description and achievements",
                    "location": "Job Location"
                }
            ],
            "education": [
                {
                    "school": "Institution Name",
                    "degree": "Degree Title",
                    "field": "Field of Study",
                    "dates": "Start Year - End Year",
                    "description": "Additional details"
                }
            ],
            "skills": [
                "Skill 1", "Skill 2", "Skill 3"
            ]
        }
        
        # Save template
        with open('linkedin_template.json', 'w') as file:
            json.dump(template, file, indent=2)
        
        print("Created linkedin_template.json - fill this out with your LinkedIn data")
        return template
    
    def extract_all_sources(self, 
                           resume_path: str = None,
                           github_username: str = None,
                           leetcode_username: str = None,
                           linkedin_data_file: str = None) -> List[Dict]:
        """Extract data from all sources"""
        
        all_chunks = []
        
        print("🔄 Starting comprehensive data extraction...")
        
        # Extract resume
        if resume_path and os.path.exists(resume_path):
            print("📄 Extracting resume data...")
            if resume_path.lower().endswith('.pdf'):
                resume_chunks = self.extract_resume_pdf(resume_path)
            elif resume_path.lower().endswith('.docx'):
                resume_chunks = self.extract_resume_docx(resume_path)
            else:
                print("❌ Unsupported resume format. Use PDF or DOCX.")
                resume_chunks = []
            
            all_chunks.extend(resume_chunks)
            print(f"✅ Extracted {len(resume_chunks)} resume chunks")
        
        # Extract GitHub
        if github_username:
            print("🐙 Extracting GitHub data...")
            github_chunks = self.extract_github_profile(github_username)
            all_chunks.extend(github_chunks)
            print(f"✅ Extracted {len(github_chunks)} GitHub chunks")
            time.sleep(1)  # Rate limiting
        
        # Extract LeetCode
        if leetcode_username:
            print("💻 Extracting LeetCode data...")
            leetcode_chunks = self.extract_leetcode_profile(leetcode_username)
            all_chunks.extend(leetcode_chunks)
            print(f"✅ Extracted {len(leetcode_chunks)} LeetCode chunks")
            time.sleep(1)  # Rate limiting
        
        # Extract LinkedIn
        if linkedin_data_file and os.path.exists(linkedin_data_file):
            print("💼 Processing LinkedIn data...")
            linkedin_chunks = self.extract_linkedin_manual(linkedin_data_file)
            all_chunks.extend(linkedin_chunks)
            print(f"✅ Extracted {len(linkedin_chunks)} LinkedIn chunks")
        elif not linkedin_data_file:
            print("💼 Creating LinkedIn template...")
            self.create_linkedin_template()
        
        print(f"🎉 Total chunks extracted: {len(all_chunks)}")
        
        # Save all extracted data
        with open('comprehensive_portfolio_data.json', 'w') as file:
            json.dump(all_chunks, file, indent=2)
        print("💾 Saved all data to comprehensive_portfolio_data.json")
        
        return all_chunks

# Usage example and main function
def main():
    extractor = ComprehensiveRAGExtractor()
    
    print("🚀 Comprehensive Portfolio RAG Data Extractor")
    print("=" * 50)
    
    # Get user inputs
    resume_path = 'data/Sanika Resume August 2025.pdf'
    github_username = 'sanikakulkarni89'
    leetcode_username = 'sanikask89'
    linkedin_file = ''
    
    # Convert empty strings to None
    resume_path = resume_path if resume_path else None
    github_username = github_username if github_username else None
    leetcode_username = leetcode_username if leetcode_username else None
    linkedin_file = linkedin_file if linkedin_file else None
    
    # Extract all data
    try:
        chunks = extractor.extract_all_sources(
            resume_path=resume_path,
            github_username=github_username,
            leetcode_username=leetcode_username,
            linkedin_data_file=linkedin_file
        )
        
        print("\n📊 Extraction Summary:")
        print(f"Total chunks: {len(chunks)}")
        
        # Show breakdown by source
        sources = {}
        for chunk in chunks:
            source = chunk.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        for source, count in sources.items():
            print(f"  {source}: {count} chunks")
        
        print("\n✅ Ready to integrate with your existing RAG system!")
        print("Next step: Combine this with your portfolio data and re-run the chunking script.")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        print("Check your inputs and try again.")

if __name__ == "__main__":
    # Install required packages
    print("📦 Required packages: pip install PyPDF2 python-docx beautifulsoup4 requests")
    main()