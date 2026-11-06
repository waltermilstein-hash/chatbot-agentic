"""
"""
import base64
from typing import Union

import fitz  # PyMuPDF
import pymongo

from pymongo import MongoClient, ReturnDocument, errors
from bson.objectid import ObjectId
import gridfs
from pymongo.errors import PyMongoError

from configyml.YamlFileManager import YamlFileManager
from framework.ConfigurationManager.ConfigurationManagerSelected import ConfigurationManagerSelected, AppConfig

config = "config/config.yml"


## test

class MongoDBManager:
    def __init__(
            self,
            collection_name: str = "default",
            config: Union['AppConfig', str] = '/config/default/config.yml'
    ):
        """
        Initialize the MongoDB manager.

        Args:
            collection_name: Collection name
            config: Can be:
                   - String: path to the config.yml (e.g., '/config/default/config.yml')
                   - AppConfig: object with configurations

        Examples:
            # Form 1: With an AppConfig object (all your current calls)
            mongo = MongoDBManager("chatbot_memory", self.config)

            # Form 2: With a string (legacy/default)
            mongo = MongoDBManager("chatbot_memory", "/config/path/config.yml")

            # Form 3: Without config (uses default)
            mongo = MongoDBManager("chatbot_memory")
        """
        try:
            # DETECT WHICH TYPE OF CONFIG IT IS
            if isinstance(config, str):
                # It's a STRING → load from file (legacy)
                app_config = self._load_config_from_file(config)
            else:
                # It's an AppConfig object → use it directly
                app_config = config

            # Now app_config is always an AppConfig object
            # Get values using whichever syntax the object supports
            self.uri = self._get_mongodb_uri(app_config)
            self.db_name = self._get_mongodb_db(app_config)

            # Connect to MongoDB
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[collection_name]
            self.config = app_config  # Keep the config as an object
            self.fs = gridfs.GridFS(self.db)
            self.collection_name = collection_name

        except errors.ConfigurationError as e:
            raise Exception(f"Configuration error while connecting to MongoDB: {str(e)}")
        except FileNotFoundError:
            raise Exception("Configuration file not found.")
        except Exception as e:
            raise Exception(f"Error initializing MongoDBManager: {str(e)}")

    def _load_config_from_file(self, config_path: str):
        """
        Load configuration from a YAML file and convert it to AppConfig.

        Args:
            config_path: Path to the config.yml file

        Returns:
            AppConfig object
        """
        try:
            # Load YAML
            yaml_manager = YamlFileManager(config_path)
            config_dict = yaml_manager.load()

            # Build AppConfig from the dictionary
            return AppConfig(config_dict=config_dict)

        except Exception as e:
            raise Exception(f"Error loading config from {config_path}: {str(e)}")

    def _get_mongodb_uri(self, app_config):
        """
        Get the MongoDB URI from the config object.
        Tries both syntaxes (dot and bracket).
        """
        try:
            # Try dot syntax
            return app_config.mongodb.uri
        except:
            try:
                # Try bracket syntax
                return app_config['mongodb']['uri']
            except:
                raise Exception("Could not get mongodb.uri from the config")

    def _get_mongodb_db(self, app_config):
        """
        Get the MongoDB DB name from the config object.
        Tries both syntaxes (dot and bracket).
        """
        try:
            # Try dot syntax
            return app_config.mongodb.db
        except:
            try:
                # Try bracket syntax
                return app_config['mongodb']['db']
            except:
                raise Exception("Could not get mongodb.db from the config")

    def get_collection(self, name: str | None = None):
        """Return the requested collection (or the default one)."""
        return self.db[name or self.collection_name]

        # (optional) have your helpers accept a collection override:

    from typing import Optional

    def find_all_dict(
            self,
            filtro: dict | None = None,
            proyeccion: dict | None = None,  # legacy support (es-ES)
            projection: dict | None = None,  # standard support (en)
            collection: str | None = None,
            sort: list | None = None,  # e.g. [("timestamp", -1)]
            limit: int | None = None  # ignored: returns 1 dict
    ) -> Optional[dict]:
        """
        Return ONE document (the first according to 'sort') as a dict, or None if there is none.
        Supports projection/proyeccion and sort. 'limit' is ignored.
        """
        col = self.get_collection(collection)

        # Unify projection
        if projection is not None and proyeccion is None:
            proyeccion = projection

        # PyMongo find_one already supports sort and projection
        if sort:
            doc = col.find_one(filtro or {}, proyeccion, sort=sort)
        else:
            doc = col.find_one(filtro or {}, proyeccion)

        return doc

    def find_all(
            self,
            filtro: dict | None = None,
            proyeccion: dict | None = None,  # legacy support (es-ES)
            projection: dict | None = None,  # standard support (en)
            collection: str | None = None,
            sort: list | None = None,  # e.g. [("timestamp", -1)]
            limit: int | None = None
    ):
        """
        Find multiple documents. Supports projection/proyeccion, sort and limit.
        Returns a list (not a cursor).
        """
        col = self.get_collection(collection)

        # Unify projection
        if projection is not None and proyeccion is None:
            proyeccion = projection

        cursor = col.find(filtro or {}, proyeccion)

        if sort:
            # PyMongo accepts list[tuple[str,int]] or list[tuple[str,pymongo.ASCENDING/DESCENDING]]
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)

        return list(cursor)

    # def find_all(self, filtro=None, proyeccion=None, collection: str | None = None):
    #     col = self.get_collection(collection)
    #     return col.find(filtro or {}, proyeccion)

    def find_all_chatbot(
            self,
            filtro: dict | None = None,
            proyeccion: dict | None = None,  # legacy support (es-ES)
            projection: dict | None = None,  # standard support (en)
            collection: str | None = None,
            sort: list | None = None,  # e.g. [("timestamp", -1)]
            limit: int | None = None
    ):
        """
        Find multiple documents. Supports projection/proyeccion, sort and limit.
        Returns a list (not a cursor).
        """
        # Unify projection
        if projection is not None and proyeccion is None:
            proyeccion = projection

        cursor = self.collection.find(filtro or {}, proyeccion)

        if sort:
            # PyMongo accepts list[tuple[str,int]] or list[tuple[str,pymongo.ASCENDING/DESCENDING]]
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)

        return list(cursor)

    def aggregate(self, pipeline, collection: str | None = None):
        col = self.get_collection(collection)
        return col.aggregate(pipeline)

    def count_documents(self, filtro=None, collection: str | None = None):
        col = self.get_collection(collection)
        return col.count_documents(filtro or {})

    def list_files(self):
        """
        List all files stored in GridFS.
        :return: List of stored files.
        """
        try:
            return self.fs.find()
        except Exception as e:
            raise Exception(f"Error listing files: {str(e)}")

    def count_documents(self, filtro):
        """
        Count the number of documents in the collection matching the given filter.

        Args:
            filtro: Dictionary representing the search condition.

        Returns:
            Integer with the number of documents matching the filter.

        Raises:
            Exception: If an error occurs during the count operation.
        """
        if filtro is None:
            filtro = {}  # Count all documents if no filter is provided

        try:
            with self.client.start_session() as session:
                with self._tx(session):
                    count = self.collection.count_documents(filtro, session=session)
                    return count
        except errors.PyMongoError as e:
            raise Exception(f"Error counting documents: {str(e)}")

    def _tx(self, session):
        """Transaction context manager TOLERANT to standalone Mongo.

        Mongo transactions require a replica set; on a standalone Mongo
        (no replica) `start_transaction()` fails. Here it is auto-detected once:
        if the deployment is a replica set a real transaction is used, otherwise a
        nullcontext (the operations still run, without a transaction)."""
        import contextlib
        if getattr(self, "_tx_supported", None) is None:
            try:
                self._tx_supported = "setName" in self.client.admin.command("hello")
            except Exception:
                self._tx_supported = False
        return session.start_transaction() if self._tx_supported else contextlib.nullcontext()

    def aggregate(self, pipeline):
        """
        Run an aggregation operation on the collection.

        Args:
            pipeline: List representing the MongoDB aggregation pipeline.

        Returns:
            List of documents resulting from the aggregation.

        Raises:
            Exception: If an error occurs during the aggregation.
        """
        if not pipeline:
            return []  # Return an empty list if the pipeline is empty

        try:
            with self.client.start_session() as session:
                with self._tx(session):
                    cursor = self.collection.aggregate(pipeline, session=session)
                    return list(cursor)
        except errors.PyMongoError as e:
            raise Exception(f"Error running the aggregation: {str(e)}")

    def insert_many(self, documents):
        """
        Insert multiple documents into the collection using a transaction.

        Args:
            documents: List of documents to insert

        Returns:
            List of strings with the IDs of the inserted documents

        Raises:
            Exception: If an error occurs during the insertion
        """
        if not documents:
            return []  # Return an empty list if there are no documents to insert

        try:
            with self.client.start_session() as session:
                with self._tx(session):
                    result = self.collection.insert_many(documents, session=session)
                    # Convert the ObjectIds to strings to return them
                    return [str(inserted_id) for inserted_id in result.inserted_ids]
        except errors.PyMongoError as e:
            raise Exception(f"Error inserting multiple documents: {str(e)}")

    def insert_one(self, document):
        """Insert a document into the collection using a transaction."""
        try:
            with self.client.start_session() as session:
                with self._tx(session):
                    result = self.collection.insert_one(document, session=session)
                    return str(result.inserted_id)
        except errors.PyMongoError as e:
            raise Exception(f"Error inserting the document: {str(e)}")

    def delete_many_by_query(self, query, collection_name):
        """Delete multiple documents from the given collection using a query."""
        try:
            # Check whether the collection exists in the database
            db = self.client[self.db.name]  # Access the database
            if collection_name not in db.list_collection_names():
                return {"message": f"The collection '{collection_name}' does not exist in the database."}

            # Get the collection by its name
            collection = db[collection_name]

            # Start the session and the transaction
            with self.client.start_session() as session:
                with self._tx(session):
                    # Delete the documents matching the query
                    result = collection.delete_many(query, session=session)

                    if result.deleted_count == 0:
                        return {
                            "message": "No documents were found matching the provided criteria."}

                    return {"message": f"{result.deleted_count} document(s) were deleted."}

        except errors.PyMongoError as e:
            return {"message": f"Error deleting the documents: {str(e)}"}

    def drop_collection(self, collection_name: str):
        """
        Drop a MongoDB collection.

        NOTE: MongoDB does not allow dropping collections inside a transaction.
        This method follows the structure of delete_many_by_query to stay consistent.
        """
        try:
            # Check whether the collection exists in the database
            db = self.client[self.db.name]
            if collection_name not in db.list_collection_names():
                return {"message": f"The collection '{collection_name}' does not exist in the database."}

            # Get the collection
            collection = db[collection_name]

            # Try to simulate a transaction (NOT allowed on drop)
            with self.client.start_session() as session:
                try:
                    # ⚠️ Cannot be done inside session.start_transaction()
                    # Therefore, it runs directly outside of a transaction.
                    collection.drop()
                    return {"message": f"The collection '{collection_name}' was deleted successfully."}

                except errors.PyMongoError as e:
                    return {"message": f"Error deleting the collection: {str(e)}"}

        except errors.PyMongoError as e:
            return {"message": f"General error deleting the collection: {str(e)}"}
    def delete_one_by_query(self, query, collection_name):
        """Delete one document from the given collection using a query."""
        try:
            # Check whether the collection exists in the database
            db = self.client[self.db.name]  # Change it for your database
            if collection_name not in db.list_collection_names():
                return {"message": f"The collection '{collection_name}' does not exist in the database."}

            # Get the collection by its name
            collection = db[collection_name]

            # Start the session and the transaction
            with self.client.start_session() as session:
                with self._tx(session):
                    # Delete the document matching the query
                    result = collection.delete_one(query, session=session)

                    if result.deleted_count == 0:
                        return {
                            "message": "No document was found matching the provided criteria."}

                    return {"message": f"{result.deleted_count} document(s) were deleted."}

        except errors.PyMongoError as e:
            return {"message": f"Error deleting the document: {str(e)}"}

    def delete_any_by_query(self, query, collection_name):
        """Delete one document from the given collection using a query."""
        try:
            # Check whether the collection exists in the database
            db = self.client[self.db.name]  # Change it for your database
            if collection_name not in db.list_collection_names():
                return {"message": f"The collection '{collection_name}' does not exist in the database."}

            # Get the collection by its name
            collection = db[collection_name]

            # Start the session and the transaction
            with self.client.start_session() as session:
                with self._tx(session):
                    # Delete the document matching the query
                    result = collection.delete_many(query, session=session)

                    if result.deleted_count == 0:
                        return {
                            "message": "No document was found matching the provided criteria."}

                    return {"message": f"{result.deleted_count} document(s) were deleted."}

        except errors.PyMongoError as e:
            return {"message": f"Error deleting the document: {str(e)}"}

    def find_and_delete(self, filter_query, verify=True):
        """
        Find and delete documents with optional verification.

        Args:
            filter_query (dict): Search filter
            verify (bool): If True, verify the deletion AFTER confirming the transaction

        Returns:
            dict: Information about the operation
        """
        try:
            # Count before the deletion
            count_before = self.collection.count_documents(filter_query)

            with self.client.start_session() as session:
                with self._tx(session):
                    result = self.collection.delete_many(filter_query, session=session)

                    if result.deleted_count == 0:
                        raise Exception("No documents were found matching the filter.")

                # The transaction was already committed automatically on leaving the context

            # Verification AFTER the transaction has been committed
            if verify:
                count_after = self.collection.count_documents(filter_query)
                if count_after != 0:
                    raise Exception(f"Verification error: {count_after} documents still exist after the commit")

            return {
                "deleted_count": result.deleted_count,
                "documents_before": count_before,
                "verified": verify,
                "success": result.deleted_count > 0
            }

        except errors.PyMongoError as e:
            raise Exception(f"Error deleting documents: {str(e)}")

    def delete_one(self, document_id):
        """Delete a document from the collection by its ID using a transaction."""
        try:
            with self.client.start_session() as session:
                with self._tx(session):
                    result = self.collection.delete_one({"_id": ObjectId(document_id)}, session=session)
                    if result.deleted_count == 0:
                        raise Exception("The document with the provided ID does not exist.")
                    return result.deleted_count
        except errors.PyMongoError as e:
            raise Exception(f"Error deleting the document: {str(e)}")

    def delete_and_archive(self, document_id, user):
        """Delete a document and archive it in the historical collection inside a transaction."""
        try:
            # Build the user's historical collection name
            historico_collection = self.db[f"{user}_historico"]

            # Start a session for the transaction
            with self.client.start_session() as session:
                # Start the transaction
                with self._tx(session):
                    # Find the document that will be deleted
                    document = self.collection.find_one({"_id": ObjectId(document_id)}, session=session)

                    if not document:
                        raise Exception("The document with the provided ID does not exist.")

                    # Delete the document from the main collection
                    result = self.collection.delete_one({"_id": ObjectId(document_id)}, session=session)

                    if result.deleted_count == 0:
                        raise Exception("Could not delete the document.")

                    # Insert the deleted document into the historical collection
                    historico_collection.insert_one(document, session=session)

                    # Return the number of deleted documents
                    return result.deleted_count

        except errors.PyMongoError as e:
            raise Exception(f"Error during the transaction: {str(e)}")

    def save_to_mongo(self, cabecera_df, name_header, detalles_df, name_details):
        """
        Save the header and detail DataFrames into MongoDB inside a transaction.
        :param db_name: MongoDB database name.
        :param cabecera_df: Header DataFrame.
        :param detalles_df: Detail DataFrame.
        """
        # Connect to MongoDB
        # Convert DataFrames to dictionaries
        cabecera_collection = self.db[name_header]
        detalles_collection = self.db[name_details]

        # Start a session for the transaction
        with self.client.start_session() as session:
            try:
                with self._tx(session):
                    # Insert the header documents
                    if not cabecera_df.empty:
                        cabecera_dict = cabecera_df.to_dict('records')
                        result = cabecera_collection.insert_many(cabecera_dict, session=session)
                        inserted_ids = result.inserted_ids  # Get the inserted IDs

                    # Insert the detail documents, associating each with a header ID
                    if not detalles_df.empty:
                        detalles_dicts = detalles_df.to_dict('records')
                        for detalle in detalles_dicts:
                            # Assign the corresponding header ID to the detail
                            detalle['relation_id'] = inserted_ids[
                                0]  # If there are several, adjust the logic to the exact relation
                        detalles_collection.insert_many(detalles_dicts, session=session)

                print("Data saved to MongoDB inside a transaction.")

            except PyMongoError as e:
                print(f"Error saving the data to MongoDB: {str(e)}")

    def update_one_new(self, document_id, update_fields):
        """Update a document in the collection by its ID, modifying only the existing fields."""
        try:
            with self.client.start_session() as session:
                with self._tx(session):
                    # Get the existing document to validate the fields
                    existing_document = self.collection.find_one({"_id": ObjectId(document_id)}, session=session)
                    if not existing_document:
                        raise Exception("The document with the provided ID does not exist.")

                    # Filter fields to update only those already present in the document
                    existing_fields = existing_document.keys()
                    filtered_update_fields = {
                        key: value for key, value in update_fields.items() if key in existing_fields
                    }

                    if not filtered_update_fields:
                        raise Exception("There are no valid fields to update in the document.")

                    # Update only the filtered fields
                    updated_document = self.collection.find_one_and_update(
                        {"_id": ObjectId(document_id)},
                        {"$set": filtered_update_fields},
                        return_document=ReturnDocument.AFTER,
                        session=session
                    )

                    return updated_document
        except errors.PyMongoError as e:
            raise Exception(f"Error updating the document: {str(e)}")

    def update_one_new_(self, document_id, update_fields):
        """Update a document in the collection by its ID using a transaction."""
        try:
            with self.client.start_session() as session:
                with self._tx(session):
                    updated_document = self.collection.find_one_and_update(
                        {"_id": ObjectId(document_id)},
                        update_fields,  # must already include $set outside
                        return_document=ReturnDocument.AFTER,
                        session=session
                    )
                    if updated_document is None:
                        raise Exception("The document with the provided ID does not exist.")
                    return updated_document
        except errors.PyMongoError as e:
            raise Exception(f"Error updating the document: {str(e)}")



    def update_one_set(self, filtro, update_fields, upsert=False):
        """
        Update a document matching the filter. If it does not exist and upsert=True, insert it.
        """
        try:
            with self.client.start_session() as session:
                with self._tx(session):
                    # ✅ Detect whether it already has MongoDB operators
                    if any(key.startswith('$') for key in update_fields.keys()):
                        update_doc = update_fields  # Already has operators, use directly
                    else:
                        update_doc = {"$set": update_fields}  # Wrap in $set

                    updated_document = self.collection.find_one_and_update(
                        filtro,
                        update_doc,
                        return_document=ReturnDocument.AFTER,
                        session=session,
                        upsert=upsert
                    )
                    if updated_document is None and not upsert:
                        raise Exception("The document with the provided filter does not exist.")
                    return updated_document
        except errors.PyMongoError as e:
            raise Exception(f"Error updating the document: {str(e)}")
    def update_one(self, filtro, update_fields, upsert=False):
        """
        Update a document matching the filter. If it does not exist and upsert=True, insert it.
        """
        try:
            with self.client.start_session() as session:
                with self._tx(session):
                    updated_document = self.collection.find_one_and_update(
                        filtro,
                        {"$set": update_fields},
                        return_document=ReturnDocument.AFTER,
                        session=session,
                        upsert=upsert
                    )
                    if updated_document is None and not upsert:
                        raise Exception("The document with the provided filter does not exist.")
                    return updated_document
        except errors.PyMongoError as e:
            raise Exception(f"Error updating the document: {str(e)}")
    def find_one(self, document_id):
        """Find a document in the collection by its ID."""
        try:
            document = self.collection.find_one({"_id": ObjectId(document_id)})
            if document is None:
                raise Exception("The document with the provided ID does not exist.")
            return document
        except errors.PyMongoError as e:
            raise Exception(f"Error searching for the document: {str(e)}")

    def find_one_query(self, query={}):
        """Find a document in the collection by its ID."""
        try:
            document = self.collection.find_one(query)
            return document
        except errors.PyMongoError as e:
            raise Exception(f"Error searching for the document: {str(e)}")

    def find_all_clasic(self, query={}):
        """Find all documents matching a query."""
        try:
            return self.collection.find(query)
        except errors.PyMongoError as e:
            raise Exception(f"Error searching for documents: {str(e)}")

    def find_all(self, query={}):
        """Find all documents matching a query."""
        try:
            documents = self.collection.find(query)
            return list(documents)
        except errors.PyMongoError as e:
            raise Exception(f"Error searching for documents: {str(e)}")

    def find_all_dict(self, query={}):
        """Find all documents matching a query."""
        try:
            documents = self.collection.find(query)
            return list(documents)
        except errors.PyMongoError as e:
            raise Exception(f"Error searching for documents: {str(e)}")

    def find_all(self, query={}, projection={}):
        """Find all documents matching a query."""
        try:
            documents = self.collection.find(query,projection)
            return list(documents)
        except errors.PyMongoError as e:
            raise Exception(f"Error searching for documents: {str(e)}")
    # GridFS methods
    def extract_pdf_pages_as_binary(self, filename: str):
        # Find the file in GridFS
        file = self.fs.find_one({"filename": filename})

        if not file:
            raise FileNotFoundError("The file was not found in GridFS.")

        # Read the PDF file from GridFS
        pdf_data = file.read()

        # Open the PDF file using PyMuPDF
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

        pages_as_binary = []

        # Extract each page as a binary image
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)  # Load the page
            # Render the page as an image
            pix = page.get_pixmap()  # Convert the page to an image
            img_binary = pix.tobytes()  # Convert the image to binary
            pages_as_binary.append(img_binary)

        return pages_as_binary

    # Usage example
    def insert_file(self, file_data, filename, is_base64=False, content_type='image/png', metadata=None):
        """Insert a file into GridFS.

        :param file_data: The file data, binary or Base64-encoded.
        :param filename: The name of the file to store.
        :param is_base64: Whether the file is Base64-encoded. If True, decode before storing.
        :param metadata: (optional) owner/context dict to store in GridFS, e.g.
            {"user": ..., "company": ...} (per-resource ownership, #14). Backward-compatible:
            if None/empty the file is left ownerless (legacy) and get-image allows it.
        :return: The ID of the file stored in GridFS.
        """
        try:
            if is_base64:
                # If the data is Base64, decode it before inserting
                file_data = base64.b64decode(file_data)

            put_kwargs = {"filename": filename, "content_type": content_type}

            # Optional ownership (#14): only metadata with real values is persisted.
            if metadata:
                clean_meta = {k: v for k, v in metadata.items() if v is not None and v != ""}
                if clean_meta:
                    put_kwargs["metadata"] = clean_meta

            # Insert the file into GridFS
            file_id = self.fs.put(file_data, **put_kwargs)
            return str(file_id)
        except (errors.PyMongoError, base64.binascii.Error) as e:
            raise Exception(f"Error inserting the file into GridFS: {str(e)}")

    def find_file(self, file_id):
        """Find a file in GridFS by its ID."""
        try:
            grid_out = self.fs.get(ObjectId(file_id))
            file_data = grid_out.read()
            return {
                "filename": grid_out.filename,
                "data": file_data,
                # Owner metadata (#14); None on legacy files without an owner.
                "metadata": getattr(grid_out, "metadata", None),
            }
        except gridfs.errors.NoFile:
            return None
        except errors.PyMongoError as e:
            raise Exception(f"Error searching for the file in GridFS: {str(e)}")

    def find_files_array(self, file_ids):
        """
        Find multiple files in GridFS by their IDs.

        :param file_ids: List of file IDs to look up in GridFS.
        :return: List of dictionaries with information about each file found.
        """
        results = []

        for file_id in file_ids:
            try:
                grid_out = self.fs.get(ObjectId(file_id))
                file_data = grid_out.read()
                results.append({
                    "filename": grid_out.filename,
                    "data": file_data
                })
            except gridfs.errors.NoFile:
                # If the file is not found, you can add None or some error identifier
                results.append({
                    "file_id": file_id,
                    "error": "File not found"
                })
            except PyMongoError as e:
                # Handle other MongoDB errors
                results.append({
                    "file_id": file_id,
                    "error": f"Error searching for the file in GridFS: {str(e)}"
                })

        return results

    def delete_files(self, file_ids):
        """Delete multiple files from GridFS by their IDs without using transactions."""
        try:
            if not isinstance(file_ids, list) or not file_ids:
                raise ValueError("A valid list of file_ids must be provided.")

            deleted_count = 0

            for file_id in file_ids:
                try:
                    self.fs.delete(ObjectId(file_id))  # 🔹 Deletion WITHOUT a transaction
                    deleted_count += 1
                except gridfs.errors.NoFile:
                    continue  # If the file does not exist, continue with the rest

            return {"message": f"{deleted_count} of {len(file_ids)} files were deleted."}

        except ValueError as ve:
            return {"message": str(ve)}
        except errors.PyMongoError as e:
            raise Exception(f"Error deleting files in GridFS: {str(e)}")

    def delete_file_not_transaction(self, file_id):
        """Delete a file from GridFS by its ID."""
        try:
            self.fs.delete(ObjectId(file_id))
            return True
        except gridfs.errors.NoFile:
            return False
        except errors.PyMongoError as e:
            raise Exception(f"Error deleting the file in GridFS: {str(e)}")
    def delete_file(self, file_id):
        """Delete a file from GridFS by its ID using a transaction."""
        try:
            with self.client.start_session() as session:
                with self._tx(session):
                    self.fs.delete(ObjectId(file_id), session=session)
                    return True
        except gridfs.errors.NoFile:
            return False
        except errors.PyMongoError as e:
            raise Exception(f"Error deleting the file in GridFS: {str(e)}")

    # Method to store base64 images with user data

    def insert_image_base64(self, base64_dic, filename, data_user):
        """Insert a Base64-encoded image into the collection using a transaction."""
        try:
            with self.client.start_session() as session:
                with self._tx(session):
                    data_user["filename"] = filename
                    data_user["data"] = base64_dic
                    data_user["content_type"] = "image"
                    result = self.collection.insert_one(data_user, session=session)
                    return str(result.inserted_id)
        except errors.PyMongoError as e:
            raise Exception(f"Error inserting the Base64 image: {str(e)}")

    def query_and_add_column_generic(self, columns=None):
        """
        Run a query in MongoDB to get the specified columns and add an additional column called 'items'.

        :param db_name: MongoDB database name.
        :param collection_name: Name of the collection where the data was stored.
        :param columns: List of column names to extract from the collection. If None, does not project any specific column.
        :return: List of documents with the required columns and the additional 'items' column.
        """
        # Connect to MongoDB
        # Define the projection based on the provided columns
        project_stage = {col: 1 for col in (columns or [])}  # Project only the specified columns
        # If 'items' is not in the provided columns, add it as an empty list or from the database
        if 'items' not in project_stage:
            project_stage['items'] = 1  # Include the document's 'items' if it already exists
        # Define the aggregation pipeline
        pipeline = [
            {
                "$project": project_stage
            }
        ]
        # Run the aggregation
        documents = self.collection.aggregate(pipeline)
        # Convert the documents to a list and return
        return list(documents)

    def replace_one(self,document_id: str, new_data: dict):
        """
        Completely replace a document in MongoDB with the provided new data.

        Args:
            document_id (str): The ID of the document to replace.
            new_data (dict): The new document that will replace the existing one.

        Returns:
            dict: Result of the replace operation.
        """
        try:
            # Convert document_id to ObjectId if valid, otherwise use it as a string
            try:
                object_id = ObjectId(document_id)
            except Exception:
                object_id = document_id

            # Replace the whole document
            result = self.collection.replace_one({"_id": object_id}, new_data, upsert=True)

            if result.upserted_id:
                return {"message": "Document inserted successfully.", "inserted_id": str(result.upserted_id)}
            elif result.modified_count > 0:
                return {"message": "Document replaced successfully."}
            else:
                return {"message": "No changes were made."}

        except Exception as e:
            return {"error": f"Error replacing the document: {str(e)}"}
