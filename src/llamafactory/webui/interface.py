# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import platform

from ..extras.misc import fix_proxy, is_env_enabled
from ..extras.packages import is_gradio_available
from .common import save_config
from .components import (
    create_chat_box,
    create_eval_tab,
    create_export_tab,
    create_footer,
    create_infer_tab,
    create_top,
    create_train_tab,
)
from .components.top import SUPPORTED_WEBUI_LANGS
from .css import CSS
from .engine import Engine


if is_gradio_available():
    import gradio as gr


WEBUI_HEAD = r"""
<script>
(function () {
    if (window.__lfBootInstalled) return;
    window.__lfBootInstalled = true;

    window.lfBoot = function () {
    const LANGS = ["en", "ro", "hu", "he", "fr", "de", "es", "pt"];

    const findLangSelect = () => {
        const selects = Array.from(document.querySelectorAll("select"));
        return selects.find((sel) => LANGS.every((code) => Array.from(sel.options).some((o) => o.value === code)));
    };

    const getLabelText = (labelEl) => {
        if (!labelEl) return "Field";
        const clone = labelEl.cloneNode(true);
        clone.querySelectorAll("button").forEach((b) => b.remove());
        return (clone.textContent || "Field").trim();
    };

    const getInfoText = (container) => {
        const candidates = [
            ".gradio-info",
            ".block-info",
            ".gr-block-info",
            ".form-text",
            "p",
            "small",
            "[class*='info']",
        ];
        for (const sel of candidates) {
            const el = container.querySelector(sel);
            const txt = (el?.textContent || "").trim();
            if (txt) return txt;
        }
        return "No help is available for this field yet.";
    };

    const getFieldTitle = (container, label) => {
        const fromLabel = getLabelText(label);
        if (fromLabel && fromLabel !== "Field") return fromLabel;
        const input = container.querySelector("input, textarea, select, button");
        if (input) {
            const aria = (input.getAttribute("aria-label") || "").trim();
            const ph = (input.getAttribute("placeholder") || "").trim();
            if (aria) return aria;
            if (ph) return ph;
        }
        return "Field";
    };

    const showHelp = (container, label) => {
        const title = getFieldTitle(container, label);
        const info = getInfoText(container);
        window.alert(`${title}\n\n${info}`);
    };

    const addLabelHelpButton = (container, label) => {
        if (!label || label.querySelector(".lf-help-q")) return false;
        const btn = document.createElement("button");
        btn.className = "lf-help-q";
        btn.type = "button";
        btn.textContent = "?";
        btn.title = "Help";
        btn.addEventListener("click", (ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            showHelp(container, label);
        });
        label.appendChild(btn);
        return true;
    };

    const addCornerHelpButton = (container, label) => {
        if (container.querySelector(".lf-help-q-corner")) return;
        container.classList.add("lf-help-host");
        const btn = document.createElement("button");
        btn.className = "lf-help-q-corner";
        btn.type = "button";
        btn.textContent = "?";
        btn.title = "Help";
        btn.addEventListener("click", (ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            showHelp(container, label);
        });
        container.appendChild(btn);
    };

    const wireHelpIcons = () => {
        const containers = document.querySelectorAll(
            [
                ".gr-block",
                ".gradio-dropdown",
                ".gradio-textbox",
                ".gradio-slider",
                ".gradio-checkbox",
                ".gradio-radio",
                ".gradio-number",
                ".form",
                ".field",
                "[class*='form']",
                "[class*='field']",
                "[class*='block']",
            ].join(", ")
        );
        containers.forEach((container) => {
            if (container.dataset.lfHelpDone === "1") return;
            const hasInteractive = !!container.querySelector("input, textarea, select, button");
            if (!hasInteractive) return;
            const label = container.querySelector("label, .block-label, [class*='label']");
            const attachedToLabel = addLabelHelpButton(container, label);
            if (!attachedToLabel) {
                addCornerHelpButton(container, label);
            }
            container.dataset.lfHelpDone = "1";
        });
    };

    const setLanguage = (code) => {
        const sel = findLangSelect();
        if (!sel) return;
        sel.value = code;
        sel.dispatchEvent(new Event("input", { bubbles: true }));
        sel.dispatchEvent(new Event("change", { bubbles: true }));
    };

    const toggleTheme = () => {
        const body = document.body;
        body.classList.toggle("dark");
    };

    const ensureZenaMenu = () => {
        const img = document.querySelector("img[alt='image'], img[alt='Zena'], .gr-image img");
        if (!img) return;
        if (document.querySelector("#lf-zena-menu")) return;

        const menu = document.createElement("div");
        menu.id = "lf-zena-menu";
        menu.className = "lf-zena-menu";
        menu.style.display = "none";
        menu.innerHTML = `
            <h4>Zena Menu</h4>
            <label>Language</label>
            <select id="lf-zena-lang">
                ${LANGS.map((x) => `<option value="${x}">${x}</option>`).join("")}
            </select>
            <button id="lf-zena-theme" type="button">Toggle Light / Dark</button>
        `;
        document.body.appendChild(menu);

        img.style.cursor = "pointer";
        img.addEventListener("click", (ev) => {
            const rect = img.getBoundingClientRect();
            menu.style.left = `${Math.round(rect.left)}px`;
            menu.style.top = `${Math.round(rect.bottom + 8)}px`;
            menu.style.display = menu.style.display === "none" ? "block" : "none";
            ev.stopPropagation();
        });

        menu.querySelector("#lf-zena-lang")?.addEventListener("change", (ev) => setLanguage(ev.target.value));
        menu.querySelector("#lf-zena-theme")?.addEventListener("click", () => toggleTheme());

        document.addEventListener("click", (ev) => {
            if (!menu.contains(ev.target) && ev.target !== img) {
                menu.style.display = "none";
            }
        });
    };

    const run = () => {
        wireHelpIcons();
        ensureZenaMenu();
    };

    run();
    const obs = new MutationObserver(() => run());
    obs.observe(document.body, { childList: true, subtree: true });
    };

    const start = () => {
        if (window.lfBoot) {
            window.lfBoot();
        }
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }

    setTimeout(start, 300);
    setTimeout(start, 1000);
})();
</script>
"""


WEBUI_JS = r"""
() => {
    if (window.lfBoot) {
        window.lfBoot();
    }
}
"""


def _get_header_avatar_path() -> str:
    candidates = [
        os.path.join("assets", "zena.png"),
        "zena_256x256.png",
        "zena.png",
        os.path.join("assets", "logo.png"),
    ]
    for image_path in candidates:
        if os.path.exists(image_path):
            return image_path

    return os.path.join("assets", "logo.png")


def create_ui(demo_mode: bool = False) -> "gr.Blocks":
    engine = Engine(demo_mode=demo_mode, pure_chat=False)
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]

    with gr.Blocks(title=f"LLaMA Factory ({hostname})", css=CSS, js=WEBUI_JS, head=WEBUI_HEAD) as demo:
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=110):
                zena_avatar = gr.Image(
                    value=_get_header_avatar_path(),
                    show_label=False,
                    interactive=False,
                    container=False,
                    height=96,
                    width=96,
                    show_download_button=False,
                    show_fullscreen_button=False,
                )
            with gr.Column(scale=8):
                title = gr.HTML()
                subtitle = gr.HTML()
        if demo_mode:
            gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")

        engine.manager.add_elems("head", {"zena_avatar": zena_avatar, "title": title, "subtitle": subtitle})
        engine.manager.add_elems("top", create_top())
        lang: gr.Dropdown = engine.manager.get_elem_by_id("top.lang")

        with gr.Tab("Train"):
            engine.manager.add_elems("train", create_train_tab(engine))

        with gr.Tab("Evaluate & Predict"):
            engine.manager.add_elems("eval", create_eval_tab(engine))

        with gr.Tab("Chat"):
            engine.manager.add_elems("infer", create_infer_tab(engine))

        if not demo_mode:
            with gr.Tab("Export"):
                engine.manager.add_elems("export", create_export_tab(engine))

        engine.manager.add_elems("footer", create_footer())
        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        demo.load(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


def create_web_demo() -> "gr.Blocks":
    engine = Engine(pure_chat=True)
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]

    with gr.Blocks(title=f"LLaMA Factory Web Demo ({hostname})", css=CSS, js=WEBUI_JS, head=WEBUI_HEAD) as demo:
        lang = gr.Dropdown(choices=SUPPORTED_WEBUI_LANGS, value="en", scale=1)
        engine.manager.add_elems("top", dict(lang=lang))

        _, _, chat_elems = create_chat_box(engine, visible=True)
        engine.manager.add_elems("infer", chat_elems)

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        demo.load(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


def run_web_ui() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")
    fix_proxy(ipv6_enabled=gradio_ipv6)
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)


def run_web_demo() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")
    fix_proxy(ipv6_enabled=gradio_ipv6)
    create_web_demo().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)
