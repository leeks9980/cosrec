# -*- coding: utf-8 -*-
"""
Multimodal AI Skin Analysis System
Using preprocessed skin images and user information for cosmetic recommendations
"""

import json
import os
from datetime import datetime
import google.generativeai as genai
from PIL import Image
import cv2

class MultimodalSkinAnalyzer:
    def __init__(self, api_key=None):
        """
        Initialize the multimodal skin analyzer
        
        Args:
            api_key (str): Google AI Studio API key
        """
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
            print("Warning: API key not provided. Only local analysis available.")
    
    def load_multimodal_data(self, json_path):
        """Load multimodal data from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None
    
    def analyze_with_ai(self, image_path, user_info, skin_analysis):
        """
        Analyze skin image using AI with user information and computer vision results
        """
        if self.model is None:
            return "Error: API key not configured"
        
        try:
            # Load and process image
            image = Image.open(image_path)
            
            # Create comprehensive prompt
            prompt = f"""
Please analyze this processed skin image and provide personalized cosmetic recommendations.

COMPUTER VISION ANALYSIS RESULTS:
- Wrinkle Score: {skin_analysis.get('wrinkle_score', 'N/A')} ({skin_analysis.get('wrinkle_status', 'N/A')})
- Pore Score: {skin_analysis.get('pore_score', 'N/A')} ({skin_analysis.get('pore_status', 'N/A')})

USER INFORMATION:
- Age: {user_info.get('age', 'Not provided')}
- Gender: {user_info.get('gender', 'Not provided')}
- Skin Type: {user_info.get('skin_type', 'Not provided')}
- Main Concerns: {user_info.get('concerns', 'Not provided')}

ANALYSIS REQUIREMENTS:
1. Detailed skin feature analysis from the image
2. Professional assessment of wrinkles and pores
3. Skin type and condition diagnosis
4. Personalized skincare routine recommendations
5. Specific product recommendations (ingredients, brands)
6. Step-by-step care methods

Please provide detailed and practical advice from a dermatologist's perspective.
Focus on Korean skincare products and methods if possible.
"""
            
            # Generate analysis using AI
            response = self.model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            return f"Error during AI analysis: {e}"
    
    def comprehensive_analysis(self, json_path):
        """
        Perform comprehensive analysis using multimodal data
        """
        # Load multimodal data
        data = self.load_multimodal_data(json_path)
        if not data:
            return None
        
        # Extract information
        image_path = data.get('image_path')
        user_info = data.get('user_info', {})
        skin_analysis = data.get('skin_analysis', {})
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
        
        # Perform AI analysis
        print("? Performing AI analysis...")
        ai_analysis = self.analyze_with_ai(image_path, user_info, skin_analysis)
        
        # Create comprehensive report
        report = self.generate_report(data, ai_analysis)
        
        return {
            'original_data': data,
            'ai_analysis': ai_analysis,
            'comprehensive_report': report,
            'image_path': image_path
        }
    
    def generate_report(self, data, ai_analysis):
        """Generate comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        user_info = data.get('user_info', {})
        skin_analysis = data.get('skin_analysis', {})
        
        report = f"""
???????????????????????????????????????????????????????????
? AI SKIN ANALYSIS COMPREHENSIVE REPORT
???????????????????????????????????????????????????????????

? Analysis Date: {timestamp}
? User Information:
   - Age: {user_info.get('age', 'N/A')}
   - Gender: {user_info.get('gender', 'N/A')}
   - Skin Type: {user_info.get('skin_type', 'N/A')}
   - Main Concerns: {user_info.get('concerns', 'N/A')}

收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收
? COMPUTER VISION ANALYSIS
收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收

? Wrinkle Analysis:
   - Score: {skin_analysis.get('wrinkle_score', 'N/A')}
   - Status: {skin_analysis.get('wrinkle_status', 'N/A')}

?? Pore Analysis:
   - Score: {skin_analysis.get('pore_score', 'N/A')}
   - Status: {skin_analysis.get('pore_status', 'N/A')}

收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收
? AI EXPERT ANALYSIS
收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收

{ai_analysis}

收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收
? COMPREHENSIVE CONCLUSION
收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收

This analysis combines:
? Computer Vision: Objective quantitative analysis
? AI Expert: Contextual interpretation and recommendations
? Personalized: Tailored to user's specific needs and concerns

?? DISCLAIMER:
This analysis is for reference only and does not replace professional medical advice.
For serious skin conditions, please consult a dermatologist.

???????????????????????????????????????????????????????????
"""
        
        return report
    
    def save_analysis_report(self, results, output_dir="analysis_reports"):
        """Save analysis report to file"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"ai_analysis_report_{timestamp}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(results['comprehensive_report'])
        
        print(f"Analysis report saved: {report_path}")
        return report_path


def interactive_analysis():
    """Interactive multimodal analysis interface"""
    print("? Multimodal AI Skin Analysis System")
    print("=" * 60)
    
    # API key input
    api_key = input("? Enter Google AI Studio API key (optional): ").strip()
    if not api_key:
        print("?? No API key provided. Only local analysis available.")
    
    # Initialize analyzer
    analyzer = MultimodalSkinAnalyzer(api_key if api_key else None)
    
    # JSON file input
    json_path = input("? Enter JSON file path: ").strip()
    
    if not os.path.exists(json_path):
        print("? JSON file not found.")
        return
    
    # Perform analysis
    print("\n? Starting comprehensive analysis...")
    print("=" * 60)
    
    results = analyzer.comprehensive_analysis(json_path)
    
    if results is None:
        print("? Analysis failed.")
        return
    
    # Display results
    print(results['comprehensive_report'])
    
    # Display image
    print("\n? Displaying analyzed image...")
    image = cv2.imread(results['image_path'])
    if image is not None:
        cv2.imshow("Analyzed Skin Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save report
    save_choice = input("\n? Save analysis report? (y/n): ").lower()
    if save_choice == 'y':
        report_path = analyzer.save_analysis_report(results)
        print(f"? Detailed report: {report_path}")


if __name__ == "__main__":
    interactive_analysis()
