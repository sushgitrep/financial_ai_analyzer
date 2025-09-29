from pathlib import Path
from utils.data_manager import DataManager
import json
from typing import List, Dict, Any
import streamlit as st
import sys
import logging
import warnings
import gdown
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class SummerisationAgent:
    """Final polished topic modeling agent"""

    def __init__(self):
        self.data_manager = DataManager()
        self.current_bank = st.session_state.get('current_bank')
        self.base_path = Path(".")        
        self.config = self.data_manager.config
        self.banks_config = self.data_manager.banks_config
        # Shared file link
        url = "https://drive.google.com/file/d/1Ba0AV692DCWdE6uov-bCcC1fWDATBfkx/view?usp=sharing"

        # Extract file ID and create direct download link
        file_id = url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"

        # Download file (will save as local json file)
        output_file = "summaries_with_speakers.json"
        gdown.download(download_url, output_file, quiet=False)
        
    def read_summaries_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Reads and parses a JSON Lines file, returning a list of dictionaries.
        """
        summaries = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        summaries.append(json.loads(line))
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_path}: {e}")
        return summaries
        
    def run(self):
        output_file = "summaries_with_speakers.json"
        all_summaries = self.read_summaries_from_file(output_file)
        if all_summaries:
            print(f"✅ Successfully read {len(all_summaries)} summaries")
        else:
            print("⚠️ No data found.")
            
                
        summaries_to_print = [
            'BAML 2Q25',
            'JPM 1Q25',
            'LB 1Q08',
            'SVB 3Q22',
            'JPM 2Q25'
                            ]
               
        
        st.subheader("📝 AI Summarization")
        if 'document_data' not in st.session_state:
            st.warning("⚠️ Please process a document first in the Preprocessing tab")
        else:
            st.info("**Ready:** Document loaded and ready for summarization")
            if st.button("📝 Generate Summary", type="primary", use_container_width=True, key="gen_summary_polished"):
                with st.spinner("Generating comprehensive summary..."):
                    import time
                    time.sleep(2)
                    st.success("✅ Summary generated!")
                    st.markdown("### 📄 Document Summary")
                    # st.markdown(f"{self.format_structured_summary(all_summaries[0], all_summaries[0]['pdf_path'])}")
                    # bank_to_filter = "LB 1Q08"
                    bank_info = self.banks_config.get("banks", {}).get(self.current_bank, {})
                    bank_to_filter = bank_info.get("bankfile")
                    filtered_summary = next(
                                        (s for s in all_summaries if s.get("bank") == bank_to_filter), None )
                    if filtered_summary:
                        st.markdown(self.format_structured_summary(filtered_summary, filtered_summary['pdf_path']))
                    else:
                        st.write(f"No summary found for bank: {bank_to_filter}")   
                # st.markdown(self.format_structured_summary(all_summaries[0], all_summaries[0]['pdf_path']))

    # def format_structured_summary(self, summary_data: Dict[str, Any], pdf_path):
    #     """
    #     Formats a structured summary dictionary for clear, point-by-point printing.
    #     Includes speaker summaries and merges points for the same speaker.
    #     """
    #     if not summary_data:
    #         print("No summary data provided.")
    #         return

    #     #Print bank name and quarter from the path
    #     print(f"Earnings Call Summary - {Path(pdf_path).stem}")

    #     # Print the executive brief first as it's a string
    #     print("## Executive Brief")
    #     print(f"- {summary_data['regulator_key_points']['executive_brief']}\n")

    #     # Iterate through the rest of the dictionary which contains lists of dicts
    #     sections = [
    #         "key_findings",
    #         "material_risks",
    #         "watch_metrics",
    #         "suggested_followups"
    #     ]

    #     for section in sections:
    #         if section in summary_data['regulator_key_points'] and isinstance(summary_data['regulator_key_points'][section], list):
    #             # Convert snake_case to Title Case for a cleaner heading
    #             heading = section.replace('_', ' ').title()
    #             print(f"## {heading}")
    #             for item in summary_data['regulator_key_points'][section]:
    #                 if isinstance(item, dict) and 'summary_point' in item and 'source_excerpt' in item:
    #                     summary_point = item['summary_point']
    #                     source_excerpt = item['source_excerpt']
    #                     print(f"- {summary_point} [\"{source_excerpt}\"]")
    #             print() # Add a newline for spacing

    #     # New section for Speaker Summaries
    #     if 'speaker_summary' in summary_data and isinstance(summary_data['speaker_summary'], dict):
    #         speaker_summary = summary_data['speaker_summary']
    #         print("## Speaker Summaries")

    #         # Print analyst points as a single paragraph
    #         print("### Analyst Points")
    #         if 'analyst_points' in speaker_summary and isinstance(speaker_summary['analyst_points'], str):
    #             print(speaker_summary['analyst_points'])
    #         print()

    #         # Print management responses, merging points by speaker name
    #         print("### Management Responses")
    #         if 'management_responses' in speaker_summary and isinstance(speaker_summary['management_responses'], list):
    #             # Use a dictionary to merge points by speaker name
    #             merged_responses = {}
    #             for response in speaker_summary['management_responses']:
    #                 if 'name' in response and 'points' in response:
    #                     speaker_name = response['name']
    #                     # Append new points to an existing list, or create a new list
    #                     if speaker_name not in merged_responses:
    #                         merged_responses[speaker_name] = []
    #                     merged_responses[speaker_name].extend(response['points'])

    #             # Print the merged responses
    #             for speaker, points in merged_responses.items():
    #                 print(f"- **{speaker}:**")
    #                 # Using a nested loop for a clean, bulleted list of points
    #                 for point in points:
    #                     print(f"  - {point}")
    #             print()


    def format_structured_summary(self, summary_data: Dict[str, Any], pdf_path):
        """
        Formats a structured summary dictionary for clear, point-by-point printing.
        Includes speaker summaries and merges points for the same speaker.
        """
        if not summary_data:
            return "No summary data provided."

        output = []

        # Add bank name and quarter from the path
        output.append(f"## Earnings Call Summary - {Path(pdf_path).stem}\n")

        # Executive brief
        output.append("### Executive Brief")
        output.append(f"- {summary_data['regulator_key_points']['executive_brief']}\n")

        # Sections
        sections = [
            "key_findings",
            "material_risks",
            "watch_metrics",
            "suggested_followups"
        ]

        for section in sections:
            if section in summary_data['regulator_key_points'] and isinstance(summary_data['regulator_key_points'][section], list):
                heading = section.replace('_', ' ').title()
                output.append(f"### {heading}")
                for item in summary_data['regulator_key_points'][section]:
                    if isinstance(item, dict) and 'summary_point' in item and 'source_excerpt' in item:
                        summary_point = item['summary_point']
                        source_excerpt = item['source_excerpt']
                        output.append(f"- {summary_point} [\"{source_excerpt}\"]")
                output.append("")

        # Speaker Summaries
        if 'speaker_summary' in summary_data and isinstance(summary_data['speaker_summary'], dict):
            speaker_summary = summary_data['speaker_summary']
            output.append("### Speaker Summaries")

            # Analyst points
            output.append("#### Analyst Points")
            if 'analyst_points' in speaker_summary and isinstance(speaker_summary['analyst_points'], str):
                output.append(speaker_summary['analyst_points'])
            output.append("")

            # Management responses
            output.append("#### Management Responses")
            if 'management_responses' in speaker_summary and isinstance(speaker_summary['management_responses'], list):
                merged_responses = {}
                for response in speaker_summary['management_responses']:
                    if 'name' in response and 'points' in response:
                        speaker_name = response['name']
                        if speaker_name not in merged_responses:
                            merged_responses[speaker_name] = []
                        merged_responses[speaker_name].extend(response['points'])

                for speaker, points in merged_responses.items():
                    output.append(f"- **{speaker}:**")
                    for point in points:
                        output.append(f"  - {point}")
                output.append("")

        return "\n".join(output)


def run_summarization_agent():
    """Entry point for topic modeling agent"""
    agent = SummerisationAgent()
    agent.run()

    