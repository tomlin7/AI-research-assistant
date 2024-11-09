from __future__ import annotations

import typing
import pandas as pd
from typing import List, Dict, Any, Optional
import concurrent.futures
import logging
import streamlit as st

if typing.TYPE_CHECKING:
    from document_search import DocumentSearchSystem

class BatchDocumentProcessor:
    def __init__(self, search_system: DocumentSearchSystem):
        """
        Initialize the batch processor with a reference to the document search system.
        
        Args:
            search_system: Instance of DocumentSearchSystem
        """
        self.search_system = search_system
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('BatchProcessor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def process_csv(self, 
                   file_path: str, 
                   title_col: str, 
                   content_col: str, 
                   metadata_cols: Optional[List[str]] = None,
                   batch_size: int = 100,
                   max_workers: int = 4) -> Dict[str, Any]:
        """
        Process documents from a CSV file in batches.
        
        Args:
            file_path: Path to the CSV file
            title_col: Name of the column containing document titles
            content_col: Name of the column containing document content
            metadata_cols: List of column names to include as metadata
            batch_size: Number of documents to process in each batch
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dict containing processing statistics
        """
        try:
            df = pd.read_csv(file_path)
            total_docs = len(df)
            self.logger.info(f"Found {total_docs} documents in CSV file")

            # Validate required columns
            if title_col not in df.columns or content_col not in df.columns:
                raise ValueError(f"Required columns {title_col} and/or {content_col} not found in CSV")

            results = {
                'successful': 0,
                'failed': 0,
                'errors': [],
                'processed_ids': []
            }

            batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                progress_bar = st.progress(0)
                status_text = st.empty()

                for batch_idx, batch_df in enumerate(batches):
                    batch_results = list(executor.map(
                        lambda row: self._process_single_document(
                            row[title_col],
                            row[content_col],
                            {col: row[col] for col in (metadata_cols or []) if col in row}
                        ),
                        batch_df.to_dict('records')
                    ))

                    for result in batch_results:
                        if result['success']:
                            results['successful'] += 1
                            results['processed_ids'].append(result['doc_id'])
                        else:
                            results['failed'] += 1
                            results['errors'].append(result['error'])

                    progress = (batch_idx + 1) / len(batches)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {results['successful']} documents successfully, {results['failed']} failed")

                progress_bar.empty()
                status_text.empty()

            self.logger.info(f"Batch processing completed. Success: {results['successful']}, Failed: {results['failed']}")
            return results

        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise

    def _process_single_document(self, 
                               title: str, 
                               content: str, 
                               metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a single document and handle any errors.
        
        Args:
            title: Document title
            content: Document content
            metadata: Optional metadata dictionary
            
        Returns:
            Dict containing processing result
        """
        try:
            doc_id = self.search_system.add_document(title, content, metadata)
            return {
                'success': True,
                'doc_id': doc_id,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'doc_id': None,
                'error': str(e)
            }

    def validate_csv(self, 
                    file_path: str, 
                    title_col: str, 
                    content_col: str, 
                    metadata_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate CSV file before processing.
        
        Args:
            file_path: Path to the CSV file
            title_col: Name of the column containing document titles
            content_col: Name of the column containing document content
            metadata_cols: List of column names to include as metadata
            
        Returns:
            Dict containing validation results
        """
        try:
            df = pd.read_csv(file_path)
            validation = {
                'valid': True,
                'errors': [],
                'stats': {
                    'total_rows': len(df),
                    'empty_titles': df[title_col].isna().sum(),
                    'empty_content': df[content_col].isna().sum()
                }
            }

            for col in [title_col, content_col]:
                if col not in df.columns:
                    validation['valid'] = False
                    validation['errors'].append(f"Required column '{col}' not found")

            if metadata_cols:
                missing_cols = [col for col in metadata_cols if col not in df.columns]
                if missing_cols:
                    validation['errors'].append(f"Metadata columns not found: {', '.join(missing_cols)}")

            if validation['stats']['empty_titles'] > 0:
                validation['errors'].append(f"Found {validation['stats']['empty_titles']} empty titles")
            if validation['stats']['empty_content'] > 0:
                validation['errors'].append(f"Found {validation['stats']['empty_content']} empty content fields")

            return validation

        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)],
                'stats': None
            }
