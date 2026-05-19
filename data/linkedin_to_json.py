import pandas as pd
import json
import zipfile
import os
from pathlib import Path
import re

class LinkedInCSVConverter:
    def __init__(self):
        self.consolidated_data = {}
        
    def extract_zip(self, zip_path: str, extract_to: str = "linkedin_extracted") -> str:
        """Extract LinkedIn ZIP file"""
        print(f"📦 Extracting {zip_path}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"✅ Extracted to {extract_to}")
        return extract_to
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names"""
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
        return df
    
    def csv_to_dict(self, csv_path: str) -> dict:
        """Convert CSV to dictionary with proper data types"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"❌ Could not read {csv_path} with any encoding")
                return {}
            
            # Clean column names
            df = self.clean_column_names(df)
            
            # Handle empty/NaN values
            df = df.fillna('')
            
            # Convert to list of dictionaries
            return df.to_dict('records')
            
        except Exception as e:
            print(f"❌ Error processing {csv_path}: {e}")
            return {}
    
    def process_profile_csv(self, csv_path: str) -> dict:
        """Process Profile.csv - basic profile information"""
        data = self.csv_to_dict(csv_path)
        if data and len(data) > 0:
            profile_info = data[0]  # Usually just one row
            return {
                'name': profile_info.get('first_name', '') + ' ' + profile_info.get('last_name', ''),
                'headline': profile_info.get('headline', ''),
                'summary': profile_info.get('summary', ''),
                'industry': profile_info.get('industry', ''),
                'location': profile_info.get('geo_location', ''),
                'email': profile_info.get('email_address', ''),
                'profile_info': profile_info
            }
        return {}
    
    def process_positions_csv(self, csv_path: str) -> list:
        """Process Positions.csv - work experience"""
        data = self.csv_to_dict(csv_path)
        positions = []
        
        for position in data:
            cleaned_position = {
                'title': position.get('title', ''),
                'company': position.get('company_name', ''),
                'description': position.get('description', ''),
                'location': position.get('location', ''),
                'started_on': position.get('started_on', ''),
                'finished_on': position.get('finished_on', ''),
                'duration': self._calculate_duration(position.get('started_on', ''), position.get('finished_on', '')),
                'raw_data': position
            }
            positions.append(cleaned_position)
        
        return positions
    
    def process_education_csv(self, csv_path: str) -> list:
        """Process Education.csv - educational background"""
        data = self.csv_to_dict(csv_path)
        education = []
        
        for edu in data:
            cleaned_edu = {
                'school': edu.get('school_name', ''),
                'degree': edu.get('degree_name', ''),
                'field_of_study': edu.get('field_of_study', ''),
                'start_date': edu.get('start_date', ''),
                'end_date': edu.get('end_date', ''),
                'grade': edu.get('grade', ''),
                'activities': edu.get('activities', ''),
                'description': edu.get('description', ''),
                'raw_data': edu
            }
            education.append(cleaned_edu)
        
        return education
    
    def process_skills_csv(self, csv_path: str) -> list:
        """Process Skills.csv - skills and endorsements"""
        data = self.csv_to_dict(csv_path)
        skills = []
        
        for skill in data:
            cleaned_skill = {
                'name': skill.get('name', ''),
                'endorsement_count': skill.get('endorsement_count', 0),
                'raw_data': skill
            }
            skills.append(cleaned_skill)
        
        return skills
    
    def process_connections_csv(self, csv_path: str) -> dict:
        """Process Connections.csv - network information"""
        data = self.csv_to_dict(csv_path)
        return {
            'total_connections': len(data),
            'connections': data[:100] if len(data) > 100 else data  # Limit for privacy/size
        }
    
    def process_recommendations_csv(self, csv_path: str) -> list:
        """Process Recommendations.csv"""
        data = self.csv_to_dict(csv_path)
        recommendations = []
        
        for rec in data:
            cleaned_rec = {
                'type': rec.get('type', ''),
                'text': rec.get('text', ''),
                'person': rec.get('first_name', '') + ' ' + rec.get('last_name', ''),
                'status': rec.get('status', ''),
                'raw_data': rec
            }
            recommendations.append(cleaned_rec)
        
        return recommendations
    
    def process_certifications_csv(self, csv_path: str) -> list:
        """Process Certifications.csv"""
        data = self.csv_to_dict(csv_path)
        certifications = []
        
        for cert in data:
            cleaned_cert = {
                'name': cert.get('name', ''),
                'authority': cert.get('authority', ''),
                'started_on': cert.get('started_on', ''),
                'finished_on': cert.get('finished_on', ''),
                'license_number': cert.get('license_number', ''),
                'url': cert.get('url', ''),
                'raw_data': cert
            }
            certifications.append(cleaned_cert)
        
        return certifications
    
    def _calculate_duration(self, start_date: str, end_date: str) -> str:
        """Calculate duration between dates"""
        if not start_date:
            return ''
        
        if not end_date or end_date.lower() in ['', 'present', 'current']:
            end_date = 'Present'
        
        return f"{start_date} - {end_date}"
    
    def process_all_csvs(self, extracted_folder: str) -> dict:
        """Process all CSV files in the extracted folder"""
        consolidated = {}
        
        # Map CSV filenames to processing functions
        csv_processors = {
            'profile.csv': ('profile', self.process_profile_csv),
            'positions.csv': ('experience', self.process_positions_csv),
            'education.csv': ('education', self.process_education_csv),
            'skills.csv': ('skills', self.process_skills_csv),
            'connections.csv': ('connections', self.process_connections_csv),
            'recommendations received.csv': ('recommendations_received', self.process_recommendations_csv),
            'recommendations given.csv': ('recommendations_given', self.process_recommendations_csv),
            'certifications.csv': ('certifications', self.process_certifications_csv),
        }
        
        # Process each CSV file
        for filename, (key, processor) in csv_processors.items():
            csv_path = os.path.join(extracted_folder, filename)
            
            # Try different case variations
            possible_paths = [
                csv_path,
                csv_path.title(),
                csv_path.upper(),
                csv_path.replace('.csv', '.CSV'),
                os.path.join(extracted_folder, filename.replace('_', ' ').title() + '.csv')
            ]
            
            found_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_file = path
                    break
            
            if found_file:
                print(f"📊 Processing {filename}...")
                try:
                    result = processor(found_file)
                    if result:
                        consolidated[key] = result
                        print(f"✅ Processed {filename} - {len(result) if isinstance(result, list) else 'success'}")
                    else:
                        print(f"⚠️ {filename} processed but returned no data")
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
            else:
                print(f"⚠️ {filename} not found")
        
        # Add any additional CSV files found
        print("\n🔍 Checking for additional CSV files...")
        for file in os.listdir(extracted_folder):
            if file.endswith('.csv') and file.lower() not in [f.lower() for f in csv_processors.keys()]:
                print(f"📊 Found additional CSV: {file}")
                csv_path = os.path.join(extracted_folder, file)
                try:
                    data = self.csv_to_dict(csv_path)
                    if data:
                        file_key = file.replace('.csv', '').lower().replace(' ', '_')
                        consolidated[file_key] = data
                        print(f"✅ Added {file} as {file_key}")
                except Exception as e:
                    print(f"❌ Could not process {file}: {e}")
        
        return consolidated
    
    def convert_linkedin_export(self, zip_path: str, output_file: str = "linkedin_data.json") -> dict:
        """Main function to convert LinkedIn ZIP export to JSON"""
        
        print("🚀 LinkedIn CSV to JSON Converter")
        print("=" * 40)
        
        # Extract ZIP
        extracted_folder = self.extract_zip(zip_path)
        
        # Process all CSVs
        print(f"\n📂 Processing CSV files from {extracted_folder}...")
        consolidated_data = self.process_all_csvs(extracted_folder)
        
        # Add metadata
        consolidated_data['_metadata'] = {
            'source': 'linkedin_export',
            'processed_at': pd.Timestamp.now().isoformat(),
            'total_sections': len(consolidated_data)
        }
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Saved consolidated data to {output_file}")
        
        # Print summary
        print(f"\n📊 Conversion Summary:")
        for key, value in consolidated_data.items():
            if key != '_metadata':
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
                elif isinstance(value, dict):
                    if 'total_connections' in value:
                        print(f"  {key}: {value['total_connections']} connections")
                    else:
                        print(f"  {key}: 1 record")
                else:
                    print(f"  {key}: processed")
        
        # Clean up extracted folder
        import shutil
        try:
            shutil.rmtree(extracted_folder)
            print(f"🗑️ Cleaned up temporary folder: {extracted_folder}")
        except:
            print(f"⚠️ Could not clean up {extracted_folder} - you can delete it manually")
        
        return consolidated_data

def main():
    converter = LinkedInCSVConverter()
    
    print("🔄 LinkedIn CSV to JSON Converter")
    print("=" * 40)
    
    # Get ZIP file path
    zip_path = 'Basic_LinkedInDataExport_08-09-2025.zip'
    
    if not os.path.exists(zip_path):
        print(f"❌ File not found: {zip_path}")
        return
    
    # Get output filename
    output_file = input("💾 Output JSON filename (press Enter for 'linkedin_data.json'): ").strip()
    if not output_file:
        output_file = "linkedin_data.json"
    
    # Convert
    try:
        data = converter.convert_linkedin_export(zip_path, output_file)
        print(f"\n✅ Successfully converted LinkedIn export!")
        print(f"📄 JSON file created: {output_file}")
        print(f"📊 Total data sections: {len(data) - 1}")  # -1 for metadata
        
        print(f"\n🔗 You can now use '{output_file}' in the comprehensive RAG extractor!")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("Please check your ZIP file and try again.")

if __name__ == "__main__":
    main()