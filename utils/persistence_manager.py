"""
Enhanced Persistence Manager for Bank-Specific Data Storage
Handles plots, data, and logs for different banks
"""

import os
import yaml
import json
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

class PersistenceManager:
    """Enhanced persistence manager with bank-specific storage"""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.plots_base_path = self.base_path / "plots"
        self.data_base_path = self.base_path / "data" / "banks" 
        self.logs_path = self.base_path / "logs"

        # Ensure base directories exist
        for path in [self.plots_base_path, self.data_base_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging with timestamps"""
        logger = logging.getLogger('persistence_manager')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            log_file = self.logs_path / f"persistence_{datetime.now().strftime('%Y%m%d')}.log"

            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def get_bank_paths(self, bank_key: str) -> Dict[str, Path]:
        """Get all relevant paths for a specific bank"""
        bank_folder = self._get_bank_folder_name(bank_key)

        paths = {
            'plots': self.plots_base_path / bank_folder,
            'data': self.data_base_path / bank_folder,
            'logs': self.logs_path / f"{bank_folder}_analysis.log"
        }

        # Ensure directories exist
        for path_key, path in paths.items():
            if path_key != 'logs':  # logs is a file, not directory
                path.mkdir(parents=True, exist_ok=True)

        return paths

    def _get_bank_folder_name(self, bank_key: str) -> str:
        """Get folder name for bank (load from config if available)"""
        try:
            config_path = self.base_path / "config" / "banks.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    banks_config = yaml.safe_load(f)

                if banks_config and 'banks' in banks_config:
                    bank_info = banks_config['banks'].get(bank_key, {})
                    return bank_info.get('folder_name', bank_key)

            return bank_key
        except Exception as e:
            self.logger.error(f"Error getting bank folder name: {e}")
            return bank_key

    def save_plot(self, bank_key: str, plot_name: str, fig, plot_type: str = 'matplotlib') -> bool:
        """Save plot/chart for specific bank"""
        try:
            paths = self.get_bank_paths(bank_key)
            plot_path = paths['plots']

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if plot_type == 'matplotlib':
                filename = f"{plot_name}_{timestamp}.png"
                filepath = plot_path / filename
                fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)  # Clean up

            elif plot_type == 'plotly':
                filename = f"{plot_name}_{timestamp}.html"
                filepath = plot_path / filename
                pio.write_html(fig, filepath)

                # Also save as PNG for thumbnails
                png_filename = f"{plot_name}_{timestamp}.png"
                png_filepath = plot_path / png_filename
                pio.write_image(fig, png_filepath, width=800, height=600)

            elif plot_type == 'wordcloud':
                filename = f"{plot_name}_{timestamp}.png"
                filepath = plot_path / filename
                fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)

            self.logger.info(f"Plot saved for {bank_key}: {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving plot for {bank_key}: {e}")
            return False

    def load_latest_plots(self, bank_key: str) -> Dict[str, List[str]]:
        """Load list of latest plots for a bank"""
        try:
            paths = self.get_bank_paths(bank_key)
            plot_path = paths['plots']

            if not plot_path.exists():
                return {}

            plots = {}

            # Group plots by type
            for file_path in plot_path.glob("*.png"):
                plot_name = file_path.stem.rsplit('_', 2)[0]  # Remove timestamp

                if plot_name not in plots:
                    plots[plot_name] = []

                plots[plot_name].append(str(file_path))

            # Sort by timestamp (latest first)
            for plot_name in plots:
                plots[plot_name].sort(reverse=True)

            return plots

        except Exception as e:
            self.logger.error(f"Error loading plots for {bank_key}: {e}")
            return {}

    def save_analysis_data(self, bank_key: str, data_type: str, data: Dict[str, Any]) -> bool:
        """Save analysis results for specific bank"""
        try:
            paths = self.get_bank_paths(bank_key)
            data_path = paths['data']

            # Add metadata
            data['metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'bank_key': bank_key,
                'data_type': data_type,
                'version': '3.0.0'
            }

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{data_type}_{timestamp}.json"
            filepath = data_path / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)

            # Also save the latest version without timestamp
            latest_filename = f"{data_type}_latest.json"
            latest_filepath = data_path / latest_filename

            with open(latest_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)

            self.logger.info(f"Analysis data saved for {bank_key}: {data_type}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving data for {bank_key}: {e}")
            return False

    def load_latest_analysis_data(self, bank_key: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Load latest analysis data for specific bank"""
        try:
            paths = self.get_bank_paths(bank_key)
            data_path = paths['data']

            latest_filepath = data_path / f"{data_type}_latest.json"

            if latest_filepath.exists():
                with open(latest_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.logger.info(f"Loaded latest {data_type} data for {bank_key}")
                return data

            return None

        except Exception as e:
            self.logger.error(f"Error loading data for {bank_key}: {e}")
            return None

    def has_existing_data(self, bank_key: str) -> Dict[str, bool]:
        """Check what data exists for a specific bank"""
        try:
            paths = self.get_bank_paths(bank_key)

            existing_data = {
                'document_data': False,
                'topic_results': False,
                'sentiment_results': False,
                'summary_results': False,
                'plots': False
            }

            # Check for data files
            data_path = paths['data']
            if data_path.exists():
                for data_type in ['document_data', 'topic_results', 'sentiment_results', 'summary_results']:
                    latest_file = data_path / f"{data_type}_latest.json"
                    if latest_file.exists():
                        existing_data[data_type] = True

            # Check for plots
            plot_path = paths['plots']
            if plot_path.exists() and any(plot_path.glob("*.png")):
                existing_data['plots'] = True

            return existing_data

        except Exception as e:
            self.logger.error(f"Error checking existing data for {bank_key}: {e}")
            return {key: False for key in ['document_data', 'topic_results', 'sentiment_results', 'summary_results', 'plots']}

    def clear_bank_data(self, bank_key: str, data_types: List[str] = None) -> bool:
        """Clear specific or all data for a bank"""
        try:
            paths = self.get_bank_paths(bank_key)

            if data_types is None:
                data_types = ['all']

            if 'all' in data_types:
                # Clear all data and plots
                for path_key, path in paths.items():
                    if path_key == 'logs':
                        continue  # Don't delete log files

                    if path.exists():
                        if path.is_dir():
                            shutil.rmtree(path)
                            path.mkdir(parents=True, exist_ok=True)
            else:
                # Clear specific data types
                data_path = paths['data']
                for data_type in data_types:
                    if data_type == 'plots':
                        plot_path = paths['plots']
                        if plot_path.exists():
                            shutil.rmtree(plot_path)
                            plot_path.mkdir(parents=True, exist_ok=True)
                    else:
                        # Clear specific data files
                        latest_file = data_path / f"{data_type}_latest.json"
                        if latest_file.exists():
                            latest_file.unlink()

                        # Also clear timestamped files
                        for file_path in data_path.glob(f"{data_type}_*.json"):
                            file_path.unlink()

            self.logger.info(f"Cleared data for {bank_key}: {data_types}")
            return True

        except Exception as e:
            self.logger.error(f"Error clearing data for {bank_key}: {e}")
            return False

    def get_bank_storage_stats(self, bank_key: str) -> Dict[str, Any]:
        """Get storage statistics for a specific bank"""
        try:
            paths = self.get_bank_paths(bank_key)

            stats = {
                'data_files': 0,
                'plot_files': 0,
                'total_size_mb': 0,
                'last_updated': None
            }

            # Count data files
            data_path = paths['data']
            if data_path.exists():
                data_files = list(data_path.glob("*.json"))
                stats['data_files'] = len(data_files)

                # Get last updated
                if data_files:
                    latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
                    stats['last_updated'] = datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()

            # Count plot files
            plot_path = paths['plots']
            if plot_path.exists():
                plot_files = list(plot_path.glob("*"))
                stats['plot_files'] = len(plot_files)

            # Calculate total size
            for path in [data_path, plot_path]:
                if path.exists():
                    for file_path in path.rglob("*"):
                        if file_path.is_file():
                            stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)

            return stats

        except Exception as e:
            self.logger.error(f"Error getting storage stats for {bank_key}: {e}")
            return {'data_files': 0, 'plot_files': 0, 'total_size_mb': 0, 'last_updated': None}

    def list_all_banks_with_data(self) -> List[Dict[str, Any]]:
        """List all banks that have stored data"""
        try:
            banks_with_data = []

            # Check data directories
            if self.data_base_path.exists():
                for bank_dir in self.data_base_path.iterdir():
                    if bank_dir.is_dir():
                        bank_key = bank_dir.name
                        existing_data = self.has_existing_data(bank_key)
                        storage_stats = self.get_bank_storage_stats(bank_key)

                        if any(existing_data.values()):
                            banks_with_data.append({
                                'bank_key': bank_key,
                                'existing_data': existing_data,
                                'storage_stats': storage_stats
                            })

            return banks_with_data

        except Exception as e:
            self.logger.error(f"Error listing banks with data: {e}")
            return []