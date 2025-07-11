import base64
import logging
import time
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import FastAPI
from gradio.themes.utils.colors import slate
from injector import inject, singleton
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.types import TokenGen
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.recipes.summarize.summarize_service import SummarizeService
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"
LOGO_PATH = THIS_DIRECTORY_RELATIVE / "sbi.png"

UI_TAB_TITLE = "Private GPT"
SOURCES_SEPARATOR = "<hr>Sources: \n"


class Modes(str, Enum):
    RAG_MODE = "RAG"
    SEARCH_MODE = "Search"
    BASIC_CHAT_MODE = "Basic"
    SUMMARIZE_MODE = "Summarize"


MODES = [
    Modes.RAG_MODE,
    Modes.SEARCH_MODE,
    Modes.BASIC_CHAT_MODE,
    Modes.SUMMARIZE_MODE,
]


class Source(BaseModel):
    file: str
    page: str
    text: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> list["Source"]:
        curated = []
        for chunk in sources:
            meta = chunk.document.doc_metadata
            file_name = meta.get("file_name", "-") if meta else "-"
            page = meta.get("page_label", "-") if meta else "-"
            curated.append(Source(file=file_name, page=page, text=chunk.text))
        return list({(s.file, s.page): s for s in curated}.values())


@singleton
class PrivateGptUi:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
        summarize_service: SummarizeService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service
        self._summarize_service = summarize_service
        self._ui_block = None
        self._selected_filename = None
        default_map = {m.value: m for m in Modes}
        self._default_mode = default_map.get(settings().ui.default_mode, Modes.RAG_MODE)
        self._system_prompt = self._get_default_system_prompt(self._default_mode)

    def _chat(self, message, history, mode: Modes, *_):
        def yield_deltas(comp: CompletionGen):
            result = ""
            for d in comp.response:
                result += d if isinstance(d, str) else d.delta or ""
                yield result
                time.sleep(0.02)
            if comp.sources:
                result += SOURCES_SEPARATOR
                s = Source.curate_sources(comp.sources)
                text = "\n".join(f"{i+1}. {v.file} (p.{v.page})" for i, v in enumerate(s))
                result += f"{text}<hr>"
            yield result

        def yield_tokens(tok: TokenGen):
            result = ""
            for t in tok:
                result += str(t)
                yield result

        def build_history():
            msgs = []
            for pair in history:
                msgs.append(ChatMessage(content=pair[0], role=MessageRole.USER))
                if len(pair) > 1 and pair[1]:
                    msgs.append(ChatMessage(
                        content=pair[1].split(SOURCES_SEPARATOR)[0],
                        role=MessageRole.ASSISTANT
                    ))
            return msgs[:20]

        new = ChatMessage(content=message, role=MessageRole.USER)
        all_msgs = [*build_history(), new]
        if self._system_prompt:
            all_msgs.insert(0, ChatMessage(content=self._system_prompt, role=MessageRole.SYSTEM))

        if mode == Modes.RAG_MODE:
            ctx = None
            if self._selected_filename:
                ids = [
                    d.doc_id for d in self._ingest_service.list_ingested()
                    if d.doc_metadata and d.doc_metadata["file_name"] == self._selected_filename
                ]
                ctx = ContextFilter(docs_ids=ids)
            stream = self._chat_service.stream_chat(all_msgs, use_context=True, context_filter=ctx)
            yield from yield_deltas(stream)

        elif mode == Modes.BASIC_CHAT_MODE:
            stream = self._chat_service.stream_chat(all_msgs, use_context=False)
            yield from yield_deltas(stream)

        elif mode == Modes.SEARCH_MODE:
            chunks = self._chunks_service.retrieve_relevant(message, limit=4, prev_next_chunks=0)
            s = Source.curate_sources(chunks)
            yield "\n".join(f"{i+1}. **{v.file} (p.{v.page})**\n{v.text}" for i, v in enumerate(s))

        elif mode == Modes.SUMMARIZE_MODE:
            ctx = None
            if self._selected_filename:
                ids = [
                    d.doc_id for d in self._ingest_service.list_ingested()
                    if d.doc_metadata and d.doc_metadata["file_name"] == self._selected_filename
                ]
                ctx = ContextFilter(docs_ids=ids)
            stream = self._summarize_service.stream_summarize(
                use_context=True,
                context_filter=ctx,
                instructions=message
            )
            yield from yield_tokens(stream)

    @staticmethod
    def _get_default_system_prompt(mode: Modes):
        return {
            Modes.RAG_MODE: settings().ui.default_query_system_prompt,
            Modes.BASIC_CHAT_MODE: settings().ui.default_chat_system_prompt,
            Modes.SUMMARIZE_MODE: settings().ui.default_summarization_system_prompt,
        }.get(mode, "")

    @staticmethod
    def _get_default_mode_explanation(mode: Modes):
        return {
            Modes.RAG_MODE: "Get contextualized answers from your files.",
            Modes.SEARCH_MODE: "Find relevant chunks fast.",
            Modes.BASIC_CHAT_MODE: "Pure LLM chat, no context.",
            Modes.SUMMARIZE_MODE: "Summarize your files.",
        }.get(mode, "")

    def _set_system_prompt(self, p): self._system_prompt = p

    def _set_explanation_mode(self, e): self._explanation_mode = e

    def _set_current_mode(self, mode: Modes):
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        self._set_explanation_mode(self._get_default_mode_explanation(mode))
        return [gr.update(placeholder=self._system_prompt), gr.update(value=self._explanation_mode)]

    def _list_ingested_files(self):
        files = {d.doc_metadata.get("file_name", "-") for d in self._ingest_service.list_ingested() if d.doc_metadata}
        return [[f] for f in files]

    def _upload_file(self, files):
        paths = [Path(f) for f in files]
        names = [p.name for p in paths]
        ids = [
            d.doc_id for d in self._ingest_service.list_ingested()
            if d.doc_metadata and d.doc_metadata["file_name"] in names
        ]
        for i in ids: self._ingest_service.delete(i)
        self._ingest_service.bulk_ingest([(p.name, p) for p in paths])

    def _delete_all_files(self):
        for d in self._ingest_service.list_ingested(): self._ingest_service.delete(d.doc_id)
        return [gr.List(self._list_ingested_files()), *[gr.Button(interactive=False)]*2, gr.Textbox("All files")]

    def _delete_selected_file(self):
        for d in self._ingest_service.list_ingested():
            if d.doc_metadata and d.doc_metadata["file_name"] == self._selected_filename:
                self._ingest_service.delete(d.doc_id)
        return [gr.List(self._list_ingested_files()), *[gr.Button(interactive=False)]*2, gr.Textbox("All files")]

    def _deselect_selected_file(self):
        self._selected_filename = None
        return [gr.Button(interactive=False), gr.Button(interactive=False), gr.Textbox("All files")]

    def _selected_a_file(self, sel: gr.SelectData):
        self._selected_filename = sel.value
        return [gr.Button(interactive=True), gr.Button(interactive=True), gr.Textbox(self._selected_filename)]

    def _build_ui_blocks(self):
        logo_data = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        with gr.Blocks(
            theme=gr.themes.Soft(primary_hue=slate),
            css=f"""
                #toggle-btn {{
                    position: absolute; left: 20px; top: 35px; background: none; border: none; font-size: 28px; color: white; cursor: pointer; z-index: 10;
                }}
                .header-bar {{
                    position: relative; width: 100%; height: 100px;
                    background-color: #0056A2; /* SBI Indigo */
                    background-image: url("data:image/png;base64,{logo_data}");
                    background-repeat: no-repeat;
                    background-position: center;
                    background-size: contain;
                }}
            """
        ) as blocks:
            blocks.load(None, None, js="""
                () => {
                    const btn = document.getElementById('toggle-btn');
                    const sidebar = document.getElementById('sidebar');
                    btn.onclick = () => {
                        sidebar.style.display = sidebar.style.display === 'none' ? 'block' : 'none';
                    };
                    return [];
                }
            """)

            with gr.Row():
                gr.HTML(f"""
                    <div class='header-bar'>
                        <button id='toggle-btn'>☰</button>
                    </div>
                """)

            with gr.Row():
                with gr.Column(scale=3, elem_id="sidebar"):
                    mode = gr.Radio([m.value for m in MODES], label="Mode", value=self._default_mode)
                    explanation = gr.Textbox(placeholder=self._get_default_mode_explanation(self._default_mode), interactive=False)
                    upload_btn = gr.UploadButton("Upload File(s)", type="filepath", file_count="multiple")
                    files_list = gr.List(self._list_ingested_files, headers=["File name"], label="Ingested Files", height=235)
                    upload_btn.upload(self._upload_file, upload_btn, files_list)
                    deselect = gr.Button("Deselect", interactive=False)
                    selected = gr.Textbox("All files", label="Selected for Query or Deletion")
                    delete_file = gr.Button("Delete selected file", interactive=False)
                    delete_all = gr.Button("Delete ALL files")
                    deselect.click(self._deselect_selected_file, outputs=[delete_file, deselect, selected])
                    files_list.select(self._selected_a_file, outputs=[delete_file, deselect, selected])
                    delete_file.click(self._delete_selected_file, outputs=[files_list, delete_file, deselect, selected])
                    delete_all.click(self._delete_all_files, outputs=[files_list, delete_file, deselect, selected])
                    sys_prompt = gr.Textbox(placeholder=self._system_prompt, label="System Prompt")
                    mode.change(self._set_current_mode, mode, [sys_prompt, explanation])
                    sys_prompt.blur(self._set_system_prompt, sys_prompt)

                with gr.Column(scale=7):
                    gr.ChatInterface(
                        self._chat,
                        chatbot=gr.Chatbot(label="SBI GPT", avatar_images=(None, AVATAR_BOT)),
                        additional_inputs=[mode, upload_btn, sys_prompt]
                    )
            gr.HTML("<footer style='text-align:center; margin-top: 20px;'>Maintained by Meghdoot</footer>")
        return blocks

    def get_ui_blocks(self):
        if not self._ui_block: self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str):
        blocks = self.get_ui_blocks()
        blocks.queue()
        gr.mount_gradio_app(app, blocks, path=path, favicon_path=AVATAR_BOT)


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)

