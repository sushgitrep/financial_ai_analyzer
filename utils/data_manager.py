"""
Enhanced Data Manager 
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import matplotlib.pyplot as plt

class DataManager:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.config = self.load_config()
        self.banks_config = self.load_banks_config()

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        dirs = ['data/banks', 'plots', 'logs', 'config']
        for dir_name in dirs:
            dir_path = self.base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        try:
            config_file = self.base_path / "config" / "config.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f) or {}
        except:
            pass
        return {
            'pdf': {'max_file_size_mb': 100},
            'preprocessing': {'remove_stopwords': True, 'lowercase': True}
        }

    def load_banks_config(self) -> Dict[str, Any]:
        try:
            banks_file = self.base_path / "config" / "banks.yaml"
            if banks_file.exists():
                with open(banks_file, 'r') as f:
                    return yaml.safe_load(f) or {}
        except:
            pass
        return {
            'banks': {
                'jp_morgan': {
                    'name': 'JPMorgan Chase & Co.',
                    'ticker': 'JPM',
                    'folder_name': 'jp_morgan',
                    'default_pdf_url': 'https://example.com/jpmorgan-annual-report.pdf'
                },
                 'goldman_sachs': {
                    'name': 'The Goldman Sachs Group, Inc.',
                    'ticker': 'GS',
                    'folder_name': 'goldman_sachs',
                    'default_pdf_url': 'https://example.com/gs-annual-report.pdf'
                }
            }
        }

    def get_bank_list(self) -> List[Dict[str, Any]]:
        banks = []
        for bank_key, bank_info in self.banks_config.get('banks', {}).items():
            banks.append({
                'key': bank_key,
                'name': bank_info.get('name', bank_key),
                'ticker': bank_info.get('ticker', ''),
                'folder_name': bank_info.get('folder_name', bank_key),
                'has_data': self._check_bank_has_data(bank_key),
                'storage_stats': self._get_storage_stats(bank_key)
            })
        return banks

    def get_bank_info(self, bank_key: str) -> Optional[Dict[str, Any]]:
        bank_info = self.banks_config.get('banks', {}).get(bank_key)
        if bank_info:
            return {
                **bank_info,
                'key': bank_key,
                'has_data': self._check_bank_has_data(bank_key),
                'existing_data': self._get_existing_data_types(bank_key),
                'storage_stats': self._get_storage_stats(bank_key)
            }
        return None

    def _check_bank_has_data(self, bank_key: str) -> bool:
        try:
            data_path = self.base_path / "data" / "banks" / bank_key
            plot_path = self.base_path / "plots" / bank_key

            has_data_files = data_path.exists() and any(data_path.glob("*.json"))
            has_plot_files = plot_path.exists() and any(plot_path.glob("*.png"))

            return has_data_files or has_plot_files
        except:
            return False

    def _get_existing_data_types(self, bank_key: str) -> Dict[str, bool]:
        data_types = ['document_data', 'topic_results', 'sentiment_results', 'summary_results']
        existing = {}

        for data_type in data_types:
            try:
                data_path = self.base_path / "data" / "banks" / bank_key / f"{data_type}_latest.json"
                existing[data_type] = data_path.exists()
            except:
                existing[data_type] = False

        return existing

    def _get_storage_stats(self, bank_key: str) -> Dict[str, Any]:
        try:
            data_path = self.base_path / "data" / "banks" / bank_key
            plot_path = self.base_path / "plots" / bank_key

            data_files = len(list(data_path.glob("*.json"))) if data_path.exists() else 0
            plot_files = len(list(plot_path.glob("*.png"))) if plot_path.exists() else 0

            # Calculate total size
            total_size = 0
            for path in [data_path, plot_path]:
                if path.exists():
                    for file_path in path.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size

            return {
                'data_files': data_files,
                'plot_files': plot_files,
                'total_size_mb': total_size / 1024 / 1024
            }
        except:
            return {'data_files': 0, 'plot_files': 0, 'total_size_mb': 0}

    def save_analysis_results(self, bank_key: str, data_type: str, results: Dict[str, Any]) -> bool:
        try:
            data_path = self.base_path / "data" / "banks" / bank_key
            data_path.mkdir(parents=True, exist_ok=True)

            filename = f"{data_type}_latest.json"
            filepath = data_path / filename

            # Also save timestamped version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_filename = f"{data_type}_{timestamp}.json"
            timestamped_filepath = data_path / timestamped_filename

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            with open(timestamped_filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            return True
        except Exception as e:
            print(f"Error saving analysis results: {e}")
            return False

    def load_analysis_results(self, bank_key: str, data_type: str) -> Optional[Dict[str, Any]]:
        try:
            data_path = self.base_path / "data" / "banks" / bank_key
            filepath = data_path / f"{data_type}_latest.json"

            if filepath.exists():
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading analysis results: {e}")
        return None

    def save_plot(self, bank_key: str, plot_name: str, fig, plot_type: str = 'matplotlib') -> bool:
        """Save plot with timestamp"""
        try:
            plot_path = self.base_path / "plots" / bank_key
            plot_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{plot_name}_{timestamp}.png"
            filepath = plot_path / filename

            if plot_type == 'matplotlib':
                fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')

            return True
        except Exception as e:
            print(f"Error saving plot: {e}")
            return False

    def get_available_plots(self, bank_key: str) -> Dict[str, List[str]]:
        try:
            plot_path = self.base_path / "plots" / bank_key
            if plot_path.exists():
                plots = {}
                for file_path in sorted(plot_path.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True):
                    # Extract plot name (remove timestamp)
                    name_parts = file_path.stem.split('_')
                    if len(name_parts) >= 2:
                        plot_name = '_'.join(name_parts[:-2]) if len(name_parts) > 2 else name_parts[0]
                    else:
                        plot_name = name_parts[0]

                    if plot_name not in plots:
                        plots[plot_name] = []
                    plots[plot_name].append(str(file_path))

                return plots
        except Exception as e:
            print(f"Error getting available plots: {e}")
        return {}

    def clear_bank_data(self, bank_key: str, data_types: List[str] = None) -> bool:
        try:
            data_path = self.base_path / "data" / "banks" / bank_key
            plot_path = self.base_path / "plots" / bank_key

            # Clear data files
            if data_path.exists():
                if data_types:
                    for data_type in data_types:
                        for pattern in [f"{data_type}_*.json"]:
                            for file_path in data_path.glob(pattern):
                                file_path.unlink()
                else:
                    for file_path in data_path.glob("*.json"):
                        file_path.unlink()

            # Clear plot files if no specific data types
            if not data_types and plot_path.exists():
                for file_path in plot_path.glob("*.png"):
                    file_path.unlink()

            return True
        except Exception as e:
            print(f"Error clearing bank data: {e}")
            return False