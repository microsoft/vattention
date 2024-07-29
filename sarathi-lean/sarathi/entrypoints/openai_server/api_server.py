import asyncio
from http import HTTPStatus
from typing import Optional

import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
import ssl
from sarathi.engine.async_llm_engine import AsyncLLMEngine
from sarathi.entrypoints.openai_server.config import OpenAIServerConfig
from sarathi.entrypoints.openai_server.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    ErrorResponse,
)
from sarathi.entrypoints.openai_server.serving_chat import OpenAIServingChat
from sarathi.entrypoints.openai_server.serving_completion import OpenAIServingCompletion
from sarathi.logger import init_logger
from sarathi.benchmark.config import ConfigParser

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion

logger = init_logger(__name__)


app = fastapi.FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.stream:
        
        async def wrapped_generator():
            async for chunk in generator:
                yield chunk
        
        t = StreamingResponse(content=wrapped_generator(), media_type="text/event-stream")
        return t
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.stream:
        t = StreamingResponse(content=generator, media_type="text/event-stream")
        return t
    else:
        return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    config_openai = OpenAIServerConfig()
    config = ConfigParser().get_config()
    # exit(0)
    if config.cluster_num_replicas == 1:
        wandb_project = config.metrics_store_wandb_project
        wandb_group = config.metrics_store_wandb_group
        wandb_run_name = config.metrics_store_wandb_run_name
    else:
        wandb_project = None
        wandb_group = None
        wandb_run_name = None

    chunk_size = None
    # exit(0)
    if config.replica_scheduler_provider == "sarathi":
        chunk_size = config.sarathi_scheduler_chunk_size
    elif config.replica_scheduler_provider == "simple_chunking":
        chunk_size = config.simple_chunking_scheduler_chunk_size
    
    # config_endpoint = 
    # engine = AsyncLLMEngine.from_system_config(
    #     config.create_system_config(), verbose=(config.log_level == "debug")
    # )

    if config.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = (
                "" if config.server_root_path is None else config.server_root_path
            )
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + config.api_key:
                return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

    logger.info(f"Launching OpenAI compatible server with config: {config}")

    served_model_names = [config.model_name]

    # engine = AsyncLLMEngine.from_system_config(
    #     config.create_system_config(), verbose=(config.log_level == "debug")
    # )
    print(config.model_block_size)
    engine = AsyncLLMEngine.from_engine_args(
            # output_dir=config.output_dir,
            # model config
            model=config.model_name,
            tokenizer=config.model_name,
            tensor_parallel_size=config.model_tensor_parallel_degree,
            pipeline_parallel_size=config.model_pipeline_parallel_degree,
            attention_backend=config.model_attention_backend,
            seed=config.seed,
            dtype="float16",
            load_format=config.model_load_format,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.model_max_model_len,
            block_size=config.model_block_size,
            # scheduler config
            scheduler_type=config.replica_scheduler_provider,
            max_num_seqs=config.replica_scheduler_max_batch_size,
            # sarathi scheduler config
            chunk_size=chunk_size,
            enable_dynamic_chunking_schedule=config.sarathi_scheduler_enable_dynamic_chunking_schedule,
            low_chunk_size=config.sarathi_scheduler_low_chunk_size,
            high_chunk_size=config.sarathi_scheduler_high_chunk_size,
            chunk_schedule_max_tokens=config.sarathi_scheduler_chunk_schedule_max_tokens,
            chunk_schedule_stages=config.sarathi_scheduler_chunk_schedule_stages,
            # vllm scheduler config
            max_num_batched_tokens=config.vllm_scheduler_max_tokens_in_batch,
            # wandb config
            write_metrics=config.write_metrics,
            enable_chrome_trace=config.write_chrome_trace,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_run_name=wandb_run_name,
            wandb_sweep_id=config.metrics_store_wandb_sweep_id,
            wandb_run_id=config.metrics_store_wandb_run_id,
            # metrics config
            enable_op_level_metrics=config.metrics_store_enable_op_level_metrics,
            enable_cpu_op_level_metrics=config.metrics_store_enable_cpu_op_level_metrics,
            enable_request_outputs=config.metrics_store_enable_request_outputs,
            keep_individual_batch_metrics=config.metrics_store_keep_individual_batch_metrics,
            # engine config
            trust_remote_code=True,

    )

    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single Sarathi instance
        model_config = asyncio.run(engine.get_model_config())

    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names,
        config.response_role,
        None,
    )
    openai_serving_completion = OpenAIServingCompletion(
        engine, model_config, served_model_names
    )

    app.root_path = config.server_root_path
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        ssl_keyfile=config.ssl_keyfile,
        ssl_certfile=config.ssl_certfile,
        ssl_ca_certs=config.ssl_ca_certs,
        ssl_cert_reqs=ssl.CERT_NONE,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,      
    )