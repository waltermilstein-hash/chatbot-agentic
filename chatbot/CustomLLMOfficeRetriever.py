"""
"""

import pandas as pd
import openpyxl
import numpy as np
import re
from datetime import datetime
import json
import weaviate
import logging
import openai
import hashlib
from typing import List, Dict, Any, Optional
from textwrap import wrap

from framework.Database.MongoDb.MongoDb import MongoDBManager
# Configuration (same as the Excel path)
from config.Constants import KEY_OPEN_AI
from framework.ConfigurationManager.ConfigurationManagerSelected import ConfigurationManagerSelected
from framework.Database.MongoDb.MongoDb import MongoDBManager
from chatbot.CustomLLMRetrieverByCelery import CustomLLMRetrieverByCelery


class OfficeDocumentIndexer:
    """
    Specialized class to process Excel, Word and plain-text files and load them
    into Weaviate, with advanced cleaning algorithms and automatic type detection.
    """

    def __init__(self, weaviate_url: str = "http://localhost:8080",
                 collection_name: str = "DocumentosExcel",
                 weaviate_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None):
        """
        Initialize the Weaviate and OpenAI connection.

        Args:
            weaviate_url: Weaviate server URL
            collection_name: Weaviate collection name
            weaviate_api_key: API key if required
            openai_api_key: OpenAI API key for embeddings
        """
        self.collection_name = collection_name
        self.weaviate_url = weaviate_url
        self.initConfig = ConfigurationManagerSelected()
        self.config = self.initConfig.load_configuration("manager_extractor")

        # Configure OpenAI if an API key is provided
        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_openai_embeddings = True
        else:
            self.use_openai_embeddings = False

        # Configure the Weaviate client
        try:
            if weaviate_api_key:
                auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)
                self.client = weaviate.Client(
                    url=weaviate_url,
                    auth_client_secret=auth_config
                )
            else:
                self.client = weaviate.Client(url=weaviate_url)

            print(f"✅ Connected to Weaviate: {weaviate_url}")
            if self.use_openai_embeddings:
                print("✅ OpenAI embeddings configured")

        except Exception as e:
            print(f"❌ Error connecting to Weaviate: {e}")
            raise















    async def insert_excel_document(
            self,
            excel_file_bytes: bytes,
            file_name: str,
            user: str,
            company: str,
            batch_size: int = 100,
            doc_type: str = "excel"
    ) -> str:
        """
        Process a full Excel file, extracting ALL content as text for split_document.
        """
        try:
            if not file_name:
                raise ValueError("A file name must be provided")

            print(f"Processing full Excel: {file_name}")
            print(f"User: {user}, Company: {company}")
            print(f"Size: {len(excel_file_bytes)} bytes")

            # Normalize the class name
            collection_name = re.sub(r"[^a-zA-Z0-9_]", "_", (company or "").upper().replace(" ", "_"))

            # Check whether the class already exists in Weaviate
            exists = await self.exists_weaviate_class(collection_name)

            # Read Excel with multiple engines
            from io import BytesIO
            import pandas as pd

            excel_file = None
            engines = ['openpyxl', 'xlrd', None]

            for engine in engines:
                try:
                    excel_buffer = BytesIO(excel_file_bytes)
                    if engine:
                        excel_file = pd.ExcelFile(excel_buffer, engine=engine)
                    else:
                        excel_file = pd.ExcelFile(excel_buffer)
                    print(f"Excel read with engine: {engine or 'default'}")
                    break
                except Exception as e:
                    print(f"Error engine {engine}: {e}")
                    continue

            if excel_file is None:
                raise ValueError("Could not read the Excel file")

            sheets = excel_file.sheet_names
            print(f"Hojas detectadas: {sheets}")

            # Extract ALL text from the Excel
            full_excel_text = self.extract_full_excel_text(excel_file_bytes, sheets, file_name)

            if not full_excel_text.strip():
                raise ValueError("Could not extract text from the Excel file")

            print(f"Total extracted text: {len(full_excel_text)} characters")

            # Full metadata
            metadata_excel = {
                'file_name': file_name,
                'user': user,
                'company': company,
                'tipo_doc': doc_type,
                'file_size': len(excel_file_bytes),
                'total_hojas': len(sheets),
                'fecha_procesamiento': datetime.now().isoformat(),
                'caracteres_extraidos': len(full_excel_text)
            }

            # Extract metadata with the LLM (text sample)
            print("Extracting metadata with the LLM...")
            try:
                text_sample = full_excel_text[:3000]  # First 3000 characters
                metadata_llm = self.extract_metadata_with_llm(text_sample)
                metadata_completa = {**metadata_excel, **metadata_llm}
            except Exception as e:
                print(f"Error extracting LLM metadata: {e}")
                metadata_completa = metadata_excel

            # Insert into MongoDB
            print("Inserting metadata into MongoDB...")
            try:
                mongo = MongoDBManager(company, self.config)
                result = mongo.insert_one(metadata_completa)
                metadata_completa.pop('_id', None)
                metadata_completa['file_id'] = result
            except Exception as e:
                print(f"MongoDB error: {e}")

            # Use split_document to generate chunks
            print("Generating chunks with split_document...")
            text_chunks = self.split_document(full_excel_text, metadata_completa)

            if not text_chunks:
                raise ValueError("Could not generate chunks from the text")

            print(f"Chunks generated with split_document: {len(text_chunks)}")

            # Create the Weaviate collection
            print("Creating collection in Weaviate...")
            train = CustomLLMRetrieverByCelery(self.config)
            class_name = await train.createCollectionRetrievalAugmentedGeneration(
                text_chunks,  # List of strings from split_document
                metadata_completa,
                train.embedding_fn_batch,
                user,
                company,
            )
            class_name = await train.createCollectionRetrievalAugmentedGeneration(
                text_chunks,  # List of strings from split_document
                metadata_completa,
                train.embedding_fn_batch,
                user,
                company,
                "BUSQUEDA_GLOBAL"
            )

            print(f"Excel processed successfully:")
            print(f"   Collection: {class_name}")
            print(f"   Chunks inserted: {len(text_chunks)}")
            print(f"   Characters processed: {len(full_excel_text)}")

            return collection_name

        except ValueError as ve:
            print(f"Validation error: {ve}")
            raise
        except Exception as e:
            print(f"Error processing Excel: {e}")
            import traceback
            print(f"Detail: {traceback.format_exc()}")
            raise

    def extract_full_excel_text(self, excel_file_bytes, sheets, file_name):
        """
        Extract ALL text from the Excel robustly and completely.
        """
        from io import BytesIO
        import pandas as pd

        full_text = []

        # Add document information
        full_text.append(f"FILE: {file_name}")
        full_text.append(f"TYPE: Excel document")
        full_text.append("=" * 50)
        full_text.append("")

        engines = ['openpyxl', 'xlrd', None]

        for sheet in sheets:
            try:
                print(f"   Extracting text from sheet: {sheet}")

                # Try reading the sheet with different methods
                df = None
                sheet_title = None

                for engine in engines:
                    try:
                        # Read the full sheet first
                        excel_buffer = BytesIO(excel_file_bytes)
                        if engine:
                            df_raw = pd.read_excel(excel_buffer, sheet_name=sheet, header=None, engine=engine)
                        else:
                            df_raw = pd.read_excel(excel_buffer, sheet_name=sheet, header=None)

                        if df_raw is not None and not df_raw.empty:
                            # Look for a title in the first rows
                            sheet_title = self.find_sheet_title(df_raw)

                            # Read using detected headers
                            fila_headers = self.detect_header_row(df_raw)

                            excel_buffer = BytesIO(excel_file_bytes)
                            if engine:
                                df = pd.read_excel(excel_buffer, sheet_name=sheet, header=fila_headers, engine=engine)
                            else:
                                df = pd.read_excel(excel_buffer, sheet_name=sheet, header=fila_headers)

                            print(f"      Read with engine {engine}: {df.shape}")
                            break

                    except Exception as e:
                        print(f"      Error engine {engine}: {e}")
                        continue

                if df is None or df.empty:
                    print(f"      Skipping empty sheet: {sheet}")
                    continue

                # Add sheet information
                full_text.append(f"SHEET: {sheet}")
                if sheet_title:
                    full_text.append(f"TITLE: {sheet_title}")
                full_text.append(f"RECORDS: {len(df)}")
                full_text.append(f"COLUMNS: {len(df.columns)}")
                full_text.append("-" * 30)
                full_text.append("")

                # Extract table content
                sheet_text = self.dataframe_to_text(df, sheet)
                full_text.append(sheet_text)
                full_text.append("")
                full_text.append("=" * 50)
                full_text.append("")

                print(f"      Extracted text: {len(sheet_text)} characters")

            except Exception as e:
                print(f"      Error processing sheet {sheet}: {e}")
                # Add at least the sheet name
                full_text.append(f"SHEET: {sheet}")
                full_text.append("ERROR: Could not process this sheet")
                full_text.append("")
                continue

        return "\n".join(full_text)

    def find_sheet_title(self, df_raw):
        """
        Look for the main title in the first rows of the sheet.
        """
        try:
            for i in range(min(5, len(df_raw))):
                fila = df_raw.iloc[i]
                for value in fila.values:
                    if pd.notna(value):
                        text = str(value).strip()
                        if len(text) > 20 and not text.upper().startswith('UNNAMED'):
                            return text
            return None
        except:
            return None

    def detect_header_row(self, df_raw):
        """
        Detect the row that contains the column headers
        """
        try:
            for i in range(min(8, len(df_raw))):
                fila = df_raw.iloc[i]
                valores_no_nulos = fila.count()
                if valores_no_nulos >= len(df_raw.columns) * 0.5:  # At least 50% of columns with data
                    return i
            return 0
        except:
            return 0

    def dataframe_to_text(self, df, nombre_hoja):
        """
        Convert a full DataFrame into structured text.
        """
        try:
            text_parts = []

            # Add column names
            text_columns = "COLUMNS: " + " | ".join([str(col) for col in df.columns])
            text_parts.append(text_columns)
            text_parts.append("")

            # Convert each row to text
            for idx, fila in df.iterrows():
                try:
                    row_data = []
                    for col, value in fila.items():
                        if pd.notna(value) and str(value).strip() and str(value).strip().lower() != 'nan':
                            valor_limpio = str(value).strip()
                            # Limit the length of very long values
                            if len(valor_limpio) > 200:
                                valor_limpio = valor_limpio[:200] + "..."
                            row_data.append(f"{col}: {valor_limpio}")

                    if row_data:
                        texto_fila = f"Record {idx + 1}: " + " | ".join(row_data)
                        text_parts.append(texto_fila)

                except Exception as e:
                    print(f"         Error converting row {idx}: {e}")
                    continue

            return "\n".join(text_parts)

        except Exception as e:
            print(f"      Error converting DataFrame to text: {e}")
            return f"ERROR: Could not convert the content of {nombre_hoja}"

    def split_document(self, text: str, metadata: dict) -> list:
        """
        Split the text into chunks and optionally inject metadata.
        """
        chunks = wrap(text, 800)  # or use a better splitter if you prefer

        enriched_chunks = []
        for chunk in chunks:
            if metadata:
                metadatos = "\n".join([f"{k}: {v}" for k, v in metadata.items() if k not in ['hojas_procesadas']])[
                    :200]  # Limit metadata
                enriched_chunks.append(f"{chunk}\n---\n{metadatos}")
            else:
                enriched_chunks.append(chunk)

        return enriched_chunks


    def process_word_document(self, word_file_bytes: bytes, file_name: str = None) -> str:
        """
        Extract text from a Word file from in-memory bytes.

        Args:
            word_file_bytes (bytes): Word file content in bytes
            file_name (str, optional): File name for logging/debug

        Returns:
            str: Full text extracted from the Word document

        Raises:
            ImportError: If python-docx is not installed
            Exception: If errors occur during processing
        """
        try:
            import docx
            from io import BytesIO

            # Create an in-memory buffer from the bytes
            word_buffer = BytesIO(word_file_bytes)

            # Load the document from the buffer
            doc = docx.Document(word_buffer)

            if file_name:
                print(f"📄 Processing Word document: {file_name}")
                print(f"📦 File size: {len(word_file_bytes)} bytes")

            full_text = []

            # Extract paragraphs
            paragraphs_count = 0
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text.strip())
                    paragraphs_count += 1

            # Extract tables if present
            tables_count = 0
            for table in doc.tables:
                tables_count += 1
                table_rows = []

                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_text:
                        table_rows.append(row_text)

                if table_rows:
                    # Add table header
                    full_text.append(f"\n--- TABLE {tables_count} ---")
                    full_text.extend(table_rows)
                    full_text.append("--- END TABLE ---\n")

            if file_name:
                print(f"   ✅ Processing completed:")
                print(f"   📝 Paragraphs extracted: {paragraphs_count}")
                print(f"   📊 Tables processed: {tables_count}")
                print(f"   📄 Total characters: {len(''.join(full_text))}")

            result = "\n".join(full_text)

            # Validate that content was extracted
            if not result.strip():
                print("⚠️  Warning: no content extracted from the Word document")

            return result

        except ImportError:
            error_msg = "⚠️  python-docx is not installed. Install with: pip install python-docx"
            print(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"❌ Error processing the Word file"
            if file_name:
                error_msg += f" '{file_name}'"
            error_msg += f": {e}"
            print(error_msg)
            raise Exception(error_msg) from e


    def process_plain_text(self, text_file_bytes: bytes, file_name: str = None,
                             encoding: str = 'utf-8') -> str:
        """
        Read a plain-text file from in-memory bytes.

        Args:
            text_file_bytes (bytes): Text file content in bytes
            file_name (str, optional): File name for logging/debug
            encoding (str): Initial encoding to try (default 'utf-8')

        Returns:
            str: Text file content

        Raises:
            Exception: If the file cannot be decoded with any encoding
        """
        try:
            if file_name:
                print(f"📄 Processing text file: {file_name}")
                print(f"📦 File size: {len(text_file_bytes)} bytes")
                print(f"🔤 Initial encoding: {encoding}")

            # Try decoding with the specified encoding
            try:
                content = text_file_bytes.decode(encoding)
                if file_name:
                    print(f"✅ File decoded successfully with encoding: {encoding}")
                return content

            except UnicodeDecodeError as e:
                print(f"⚠️  Error with encoding {encoding}: {e}")

                # Try different common encodings
                encodings_alternativos = [
                    'latin-1', 'cp1252', 'iso-8859-1', 'utf-16', 'utf-32',
                    'cp850', 'cp437', 'ascii', 'windows-1252'
                ]

                for enc in encodings_alternativos:
                    try:
                        content = text_file_bytes.decode(enc)
                        print(f"✅ File decoded with alternative encoding: {enc}")
                        if file_name:
                            print(f"   📝 Content extracted: {len(content)} characters")
                        return content
                    except (UnicodeDecodeError, LookupError):
                        continue

                # If no encoding works, try errors='replace'
                print("⚠️  Trying decoding with character replacement...")
                try:
                    content = text_file_bytes.decode('utf-8', errors='replace')
                    print("✅ File decoded by replacing problematic characters")
                    return content
                except:
                    pass

                # Last resort: use chardet if available
                try:
                    import chardet
                    result = chardet.detect(text_file_bytes)
                    encoding_detectado = result['encoding']
                    confianza = result['confidence']

                    if encoding_detectado and confianza > 0.7:
                        print(
                            f"🔍 Encoding auto-detected: {encoding_detectado} (confidence: {confianza:.2f})")
                        content = text_file_bytes.decode(encoding_detectado)
                        return content
                    else:
                        print(f"⚠️  Encoding detected with low confidence: {encoding_detectado} ({confianza:.2f})")

                except ImportError:
                    print("💡 Tip: install 'chardet' for automatic encoding detection: pip install chardet")
                except:
                    pass

                error_msg = f"❌ Could not determine the file encoding"
                if file_name:
                    error_msg += f" '{file_name}'"
                print(error_msg)
                raise Exception(error_msg)

        except Exception as e:
            error_msg = f"❌ Error processing the text file"
            if file_name:
                error_msg += f" '{file_name}'"
            error_msg += f": {e}"
            print(error_msg)
            raise Exception(error_msg) from e




    async def insert_document_universal_bytes(
            self,
            file_bytes: bytes,
            file_name: str,
            user: str,
            company: str,
            batch_size: int = 100,
            doc_type: str = None
    ) -> str:
        """
        Process any document type from bytes (Excel, Word, Text) and insert it into MongoDB and Weaviate.

        Args:
            file_bytes (bytes): File content in bytes
            file_name (str): File name with extension
            user (str): User uploading the document
            company (str): Company associated with the document
            batch_size (int): Batch size for processing
            doc_type (str): Document type (auto-detected if None)

        Returns:
            str: Name of the class/collection created in Weaviate
        """
        try:
            # Initial validations
            if not file_bytes:
                raise ValueError("The file content cannot be empty")

            if not file_name:
                raise ValueError("A file name must be provided")

            # Auto-detect type if not specified
            if not doc_type:
                extension = file_name.lower().split('.')[-1] if '.' in file_name else ''
                doc_type = self.detect_file_type_bytes(file_bytes, extension)

            print(f"🚀 Processing {doc_type.upper()} file from bytes: {file_name}")
            print(f"📦 File size: {len(file_bytes)} bytes")

            # Normalize the class name
            collection_name = re.sub(r"[^a-zA-Z0-9_]", "_", (company or "").upper().replace(" ", "_"))

            # Check whether the class already exists in Weaviate
            exists = await self.exists_weaviate_class(collection_name)

            # Process according to the file type
            if doc_type == 'excel':
                return await self.insert_excel_document(
                    file_bytes,
                    file_name,
                    user,
                    company,
                    batch_size,
                    doc_type
                )

            elif doc_type == 'word':
                # Process the Word file from bytes
                print("📝 Extracting text from Word from bytes...")
                extracted_text = self.process_word_document(file_bytes, file_name)

                if not extracted_text.strip():
                    raise ValueError("Could not extract valid text from the Word file")

            elif doc_type == 'texto':
                # Process the plain-text file from bytes
                print("📄 Reading text file from bytes...")
                extracted_text = self.process_plain_text(file_bytes, file_name)

                if not extracted_text.strip():
                    raise ValueError("Could not read valid content from the text file")

            else:
                raise ValueError(f"Unsupported file type: {doc_type}")

            # For Word and text: process with the LLM and split_document
            if doc_type in ['word', 'texto']:
                # Base metadata
                metadata_base = {
                    'file_name': file_name,
                    'user': user,
                    'company': company,
                    'tipo_doc': doc_type,
                    'file_size': len(file_bytes),
                    'fecha_procesamiento': datetime.now().isoformat(),
                    'longitud_texto': len(extracted_text)
                }

                # Extract metadata with the LLM (limit text for efficiency)
                print("🤖 Extracting metadata with the LLM...")
                text_for_llm = extracted_text[:3000]  # Limit for the LLM
                metadata_llm = self.extract_metadata_with_llm(text_for_llm)

                # Merge metadata
                metadata_completa = {**metadata_base, **metadata_llm}

                # Initialize configuration

                # Insert into MongoDB
                print("💾 Inserting metadata into MongoDB...")
                mongo = MongoDBManager(company, self.config)
                result = mongo.insert_one(metadata_completa)
                metadata_completa.pop('_id', None)
                metadata_completa['file_id'] = result

                # Generate chunks using split_document
                print("✂️  Generating chunks with split_document...")
                text_chunks = self.split_document(extracted_text, metadata_completa)

                if not text_chunks:
                    raise ValueError("Could not generate chunks from the text")

                print(f"📦 {len(text_chunks)} chunks generated")

                # Create in Weaviate
                print("🔄 Creating collection in Weaviate...")
                train = CustomLLMRetrieverByCelery(self.config)
                class_name = await train.createCollectionRetrievalAugmentedGeneration(
                    text_chunks,  # ← Chunks from split_document
                    metadata_completa,
                    train.embedding_fn_batch,
                    user,
                    company,
                )
                class_name = await train.createCollectionRetrievalAugmentedGeneration(
                    text_chunks,  # ← Chunks from split_document
                    metadata_completa,
                    train.embedding_fn_batch,
                    user,
                    company,
                    "BUSQUEDA_GLOBAL"
                )
                print(f"✅ {doc_type.capitalize()} processed successfully in collection: {class_name}")
                print(f"   📦 {len(text_chunks)} chunks inserted")
                print(f"   📄 Text processed: {len(extracted_text)} characters")

                return collection_name

        except ValueError as ve:
            print(f"[Validation] Error processing {doc_type}: {ve}")
            raise
        except Exception as e:
            print(f"[Error] Error during insertion of {doc_type}: {e}")
            import traceback
            print(f"Detail: {traceback.format_exc()}")
            raise

    def detect_file_type_bytes(self, file_bytes: bytes, extension: str = "") -> str:
        """
        Detect the file type based on the byte content and the extension.

        Args:
            file_bytes (bytes): File content
            extension (str): File extension (without the dot)

        Returns:
            str: Detected type ('excel', 'word', 'texto')
        """
        try:
            print(f"🔍 Detecting file type...")
            print(f"   📁 Extension: {extension}")
            print(f"   📦 Size: {len(file_bytes)} bytes")

            # First try by known extension
            if extension in ['xlsx', 'xls', 'xlsm']:
                print("   ✅ Type detected by extension: Excel")
                return 'excel'
            elif extension in ['docx', 'doc']:
                print("   ✅ Type detected by extension: Word")
                return 'word'
            elif extension in ['txt', 'text', 'log', 'md', 'py', 'js', 'html', 'css', 'json', 'xml', 'csv', 'tsv']:
                print("   ✅ Type detected by extension: Text")
                return 'texto'

            # If there is no clear extension, use magic bytes
            if len(file_bytes) < 8:
                print("   ⚠️  File too small, assuming text")
                return 'texto'

            # Magic bytes for different formats
            magic_bytes = file_bytes[:8]

            # ZIP files (includes modern .xlsx and .docx)
            if file_bytes.startswith(b'PK\x03\x04'):
                print("   🔍 Magic bytes: ZIP file detected")

                # Look for specific markers in the first 2KB
                initial_content = file_bytes[:2048].lower()

                if b'xl/' in initial_content or b'worksheets' in initial_content or b'sharedstrings' in initial_content:
                    print("   ✅ Type detected by content: Excel (.xlsx)")
                    return 'excel'
                elif b'word/' in initial_content or b'document.xml' in initial_content:
                    print("   ✅ Type detected by content: Word (.docx)")
                    return 'word'
                else:
                    print("   ⚠️  ZIP without clear indicators, assuming Excel")
                    return 'excel'

            # OLE2 files (legacy .xls, .doc formats)
            elif file_bytes.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
                print("   🔍 Magic bytes: OLE2 file detected")

                # .xls files usually have these bytes further in
                if b'Microsoft Excel' in file_bytes[:1024] or b'Workbook' in file_bytes[:1024]:
                    print("   ✅ Type detected by content: Excel (.xls)")
                    return 'excel'
                else:
                    print("   ✅ Type detected by magic bytes: Word (.doc)")
                    return 'word'

            # RTF files
            elif file_bytes.startswith(b'{\\rtf'):
                print("   ✅ Type detected by magic bytes: Word (RTF)")
                return 'word'

            # If it doesn't match binary formats, test whether it's text
            else:
                print("   🔍 No known magic bytes detected, trying as text...")

                # Try decoding as text
                try:
                    # Try different encodings
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            sample_text = file_bytes[:500].decode(encoding)
                            # If it has many control characters, it's probably not text
                            characters_control = sum(1 for c in sample_text if ord(c) < 32 and c not in '\n\r\t')
                            if characters_control < len(sample_text) * 0.1:  # Less than 10% control characters
                                print(f"   ✅ Type detected by decoding: Text ({encoding})")
                                return 'texto'
                        except UnicodeDecodeError:
                            continue

                    print("   ⚠️  Could not decode as text, assuming unknown binary")
                    return 'texto'  # Fallback

                except Exception as e:
                    print(f"   ⚠️  Detection error: {e}, assuming text")
                    return 'texto'

        except Exception as e:
            print(f"   ❌ Error detecting file type: {e}")
            print(f"   ⚠️  Fallback: assuming plain text")
            return 'texto'
    # Excel-specific helper from bytes

    def extract_metadata_with_llm(self, text: str) -> dict:
        """
        Use an LLM to extract structured metadata from text.
        """
        import openai  # or any provider

        prompt = f"""
        Extract the relevant metadata from the following document. Return it as valid JSON and only JSON. Do not add any extra information. 
        and detect the file type,
        and return a doc_type field in the JSON with that result.
        Document text:
        {text[:3000]}  # Truncated if too long
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-5.6",
                messages=[{"role": "user", "content": prompt}],
                # gpt-5.6 (reasoner) only accepts the default temperature; passing 0 returns 400.
            )

            import json
            response_ = json.loads(
                response.choices[0].message.content.strip().replace("```json", "").replace("```", ""))
            return response_
        except Exception as e:
            print(f"Error in extract_metadata_with_llm: {e}")
            return {}  # if the LLM does not respond with valid JSON

    async def exists_weaviate_class(self, class_name: str) -> bool:
        """
        Check whether a class exists in Weaviate
        """
        try:
            return self.client.schema.exists(class_name)
        except Exception as e:
            print(f"Error checking class in Weaviate: {e}")
            return False


# Universal entry point for any document type from bytes
async def run_document_to_weaviate(
        file_bytes: bytes,
        file_name: str,
        user: str,
        company: str,
        weaviate_url: str = "http://localhost:8086",
        openai_api_key: str = None,
        batch_size: int = 100,
        doc_type: str = None,  # Auto-detect if None
) -> str:
    """
    Universal entry point to process any file from bytes and insert it into Weaviate.

    Args:
        file_bytes (bytes): File content in bytes (Excel, Word, Text)
        file_name (str): File name with extension (for type detection)
        user (str): User uploading the document
        company (str): Company associated with the document
        weaviate_url (str): Weaviate server URL
        openai_api_key (str): OpenAI API key for embeddings
        batch_size (int): Batch size for processing
        doc_type (str): Document type (auto-detected if None)

    Returns:
        str: Name of the collection created in Weaviate

    Example:
        # From an uploaded file (Flask/FastAPI)
        file_bytes = uploaded_file.read()
        collection = await run_document_to_weaviate(
            file_bytes=file_bytes,
            file_name="empleados.xlsx",
            user="admin@company.com",
            company="GENAI",
        )

        # From a file on disk
        with open("manual.docx", "rb") as f:
            file_bytes = f.read()

        collection = await run_document_to_weaviate(
            file_bytes=file_bytes,
            file_name="manual.docx",
            user="admin@company.com",
            company="GENAI",
        )

        # Text file
        with open("politicas.txt", "rb") as f:
            file_bytes = f.read()

        collection = await run_document_to_weaviate(
            file_bytes=file_bytes,
            file_name="politicas.txt",
            user="admin@company.com",
            company="GENAI",
        )
    """
    try:
        # Initial validations
        if not file_bytes:
            raise ValueError("The file content cannot be empty")

        if not file_name:
            raise ValueError("A file name with extension must be provided")

        print(f"🚀 Starting universal document processing: {file_name}")
        print(f"📦 File size: {len(file_bytes)} bytes")
        print(f"🏷️  User: {user}, Company: {company}")

        # Auto-detect the document type if not specified
        if not doc_type:
            extension = file_name.lower().split('.')[-1] if '.' in file_name else ''
            doc_type = detect_document_type(extension, file_bytes)
            print(f"🔍 Detected document type: {doc_type}")

        # Create the processor instance
        processor = OfficeDocumentIndexer(
            weaviate_url=weaviate_url,
            collection_name=re.sub(r"[^a-zA-Z0-9_]", "_", (company or "").upper().replace(" ", "_")),
            openai_api_key=openai_api_key
        )

        # Process and insert the document according to its type
        collection_name = await processor.insert_document_universal_bytes(
            file_bytes=file_bytes,
            file_name=file_name,
            user=user,
            company=company,
            batch_size=batch_size,
            doc_type=doc_type
        )

        print(f"🎉 Processing successful: {collection_name}")
        return collection_name

    except ValueError as ve:
        print(f"❌ Validation error: {ve}")
        raise
    except Exception as e:
        print(f"❌ Error in run_document_to_weaviate: {e}")
        import traceback
        print(f"Detail: {traceback.format_exc()}")
        raise


def detect_document_type(extension: str, file_bytes: bytes) -> str:
    """
    Detect the document type based on extension and content.

    Args:
        extension (str): File extension (without the dot)
        file_bytes (bytes): File content for validation

    Returns:
        str: Detected document type
    """
    try:
        # Detection by extension
        if extension in ['xlsx', 'xls']:
            return 'excel'
        elif extension in ['docx', 'doc']:
            return 'word'
        elif extension in ['txt', 'text', 'log', 'md', 'py', 'js', 'html', 'css', 'json', 'xml', 'csv']:
            return 'texto'

        # Detection by magic bytes if the extension is unclear
        if file_bytes:
            # Magic bytes for different formats
            if file_bytes.startswith(b'PK\x03\x04'):
                # ZIP file (includes .xlsx and .docx)
                if b'xl/' in file_bytes[:1000] or b'worksheets' in file_bytes[:2000]:
                    return 'excel'
                elif b'word/' in file_bytes[:1000] or b'document.xml' in file_bytes[:2000]:
                    return 'word'
            elif file_bytes.startswith(b'\xd0\xcf\x11\xe0'):
                # OLE2 format (legacy .xls and .doc files)
                return 'excel'  # Default, but it could also be word
            else:
                # Try decoding as text
                try:
                    file_bytes.decode('utf-8')
                    return 'texto'
                except:
                    try:
                        file_bytes.decode('latin-1')
                        return 'texto'
                    except:
                        pass

        # Fallback for unknown extension
        print(f"⚠️  Extension '{extension}' not recognized, assuming plain text")
        return 'texto'

    except Exception as e:
        print(f"⚠️  Error detecting document type: {e}, assuming text")
        return 'texto'
