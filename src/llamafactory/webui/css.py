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

CSS = r"""
.duplicate-button {
  margin: auto !important;
  color: white !important;
  background: black !important;
  border-radius: 100vh !important;
}

.thinking-summary {
  padding: 8px !important;
}

.thinking-summary span {
  border-radius: 4px !important;
  padding: 4px !important;
  cursor: pointer !important;
  font-size: 14px !important;
  background: rgb(245, 245, 245) !important;
}

.dark .thinking-summary span {
  background: rgb(73, 73, 73) !important;
}

.thinking-container {
  border-left: 2px solid #a6a6a6 !important;
  padding-left: 10px !important;
  margin: 4px 0 !important;
}

.thinking-container p {
  color: #a6a6a6 !important;
}

.modal-box {
  position: fixed !important;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%); /* center horizontally */
  max-width: 1000px;
  max-height: 750px;
  overflow-y: auto;
  background-color: var(--input-background-fill);
  flex-wrap: nowrap !important;
  border: 2px solid black !important;
  z-index: 1000;
  padding: 10px;
}

.dark .modal-box {
  border: 2px solid white !important;
}

.lf-help-q {
  margin-left: 6px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 1px solid #64748b;
  background: transparent;
  color: #64748b;
  font-size: 11px;
  font-weight: 700;
  line-height: 16px;
  cursor: pointer;
}

.lf-help-q:hover {
  border-color: #0ea5e9;
  color: #0ea5e9;
}

.lf-help-host {
  position: relative;
}

.lf-help-q-corner {
  position: absolute;
  top: 6px;
  right: 6px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 1px solid #64748b;
  background: rgba(17, 24, 39, 0.85);
  color: #cbd5e1;
  font-size: 11px;
  font-weight: 700;
  line-height: 16px;
  cursor: pointer;
  z-index: 10;
}

.lf-help-q-corner:hover {
  border-color: #0ea5e9;
  color: #0ea5e9;
}

.lf-zena-menu {
  position: fixed;
  z-index: 3000;
  min-width: 220px;
  border-radius: 10px;
  border: 1px solid #334155;
  background: #111827;
  color: #e5e7eb;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
  padding: 10px;
}

.lf-zena-menu h4 {
  margin: 0 0 10px 0;
  font-size: 13px;
}

.lf-zena-menu label {
  display: block;
  font-size: 12px;
  margin-bottom: 4px;
}

.lf-zena-menu select,
.lf-zena-menu button {
  width: 100%;
  margin-bottom: 8px;
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid #475569;
  background: #1f2937;
  color: #e5e7eb;
}
"""
