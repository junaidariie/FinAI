from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
from fastapi.responses import StreamingResponse
import json
import os
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from scripts.rag import RagPipeline
from scripts.main import set_rag_instance
from scripts.main import stream_chat_response


app = FastAPI(version='1.0', title='FinAI', description="A finetuned qwen model for financial QA with rag.")
rag = RagPipeline()
set_rag_instance(rag)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


@app.get('/')
def health_check():
    return {'status' : 'The api is live.'}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files allowed."
            )

        os.makedirs("temp_docs", exist_ok=True)

        file_path = f"temp_docs/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        rag.delete_all_docs()

        docs = rag.load_docs(file_path)
        split_docs = rag.split_docs(docs)

        rag.add_docs(split_docs)
        rag.create_bm25(split_docs)

        return {
            "status": "success",
            "message": "Document uploaded successfully",
            "chunks": len(split_docs),
            "document": file.filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    

@app.delete("/reset")
async def reset_docs():
    try:
        rag.delete_all_docs()

        return {
            "status": "success",
            "message": "All docs deleted"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )



@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):

    def event_generator():
        try:
            metadata_sent = False

            for chunk in stream_chat_response(
                user_message=request.message,
                thread_id=request.thread_id
            ):
                if not metadata_sent:
                    metadata_event = {
                        "type": "metadata",
                        "used_rag": chunk["metadata"]["used_rag"],
                        "sources": chunk["metadata"]["sources"],
                        "thread_id": chunk["metadata"]["thread_id"]
                    }

                    yield f"data: {json.dumps(metadata_event)}\n\n"

                    metadata_sent = True

                token_event = {
                    "type": "token",
                    "content": chunk["token"]
                }

                yield f"data: {json.dumps(token_event)}\n\n"

            done_event = {
                "type": "done"
            }

            yield f"data: {json.dumps(done_event)}\n\n"

        except Exception as e:
            error_event = {
                "type": "error",
                "message": str(e)
            }

            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

