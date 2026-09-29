/* AINode Command Center — chat: binding, model select, send/stream, message rendering.
 *
 * Methods of the AINode object, split out of app.js by view (W2 in
 * upgrade-fixes.md) so that work on one view stops conflicting with work
 * on every other. A classic script loaded after app.js: the methods are
 * the same, called the same way, with the same `this`.
 */
Object.assign(AINode, {
  // ========================================================================
  //  CHAT — BINDING (Left Panel + Bottom Bar)
  // ========================================================================

  bindChat() {
    var self = this;
    var input = document.getElementById('chat-input');
    var send = document.getElementById('chat-send');

    if (send) {
      send.addEventListener('click', function () { self.handleSendClick(); });
    }
    if (input) {
      input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); self.handleSendClick(); }
      });
      input.addEventListener('input', function () {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 200) + 'px';
      });
    }

    // New Chat button
    var newChatBtn = document.getElementById('new-chat');
    if (newChatBtn) {
      newChatBtn.addEventListener('click', function () { self.newConversation(); });
    }

    // Search conversations
    var searchInput = document.getElementById('chat-search');
    if (searchInput) {
      searchInput.addEventListener('input', function () { self.renderConversationList(); });
    }

    // Drag-and-drop image support on the center-stage area
    var stage = document.getElementById('center-stage');
    if (stage) {
      stage.addEventListener('dragover', function (e) {
        if (e.dataTransfer && Array.from(e.dataTransfer.items || []).some(function (i) { return i.kind === 'file'; })) {
          e.preventDefault();
          stage.classList.add('drag-over');
        }
      });
      stage.addEventListener('dragleave', function (e) {
        // Remove overlay when leaving the stage OR any child boundary
        if (!stage.contains(e.relatedTarget)) stage.classList.remove('drag-over');
      });
      stage.addEventListener('drop', function (e) {
        e.preventDefault();
        stage.classList.remove('drag-over');
        var files = Array.from(e.dataTransfer.files || []).filter(function (f) {
          return f.type.startsWith('image/');
        });
        if (files.length > 0) self.attachImages(files);
        else self.toast('Only image files are supported', 'warning');
      });
      // Escape key dismisses the drop overlay
      document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') stage.classList.remove('drag-over');
      });
      // Click outside the drop zone also dismisses it
      stage.addEventListener('click', function () {
        stage.classList.remove('drag-over');
      });
    }

    // Also support paste of images
    if (input) {
      input.addEventListener('paste', function (e) {
        var items = Array.from((e.clipboardData || {}).items || []);
        var imgs = items.filter(function (i) { return i.type.startsWith('image/'); }).map(function (i) { return i.getAsFile(); }).filter(Boolean);
        if (imgs.length > 0) {
          e.preventDefault();
          self.attachImages(imgs);
        }
      });
    }
  },

  attachImages(files) {
    var self = this;
    if (!this.state.pendingAttachments) this.state.pendingAttachments = [];
    files.forEach(function (file) {
      if (file.size > 10 * 1024 * 1024) {
        self.toast(file.name + ' is too large (max 10 MB)', 'error');
        return;
      }
      var reader = new FileReader();
      reader.onload = function (e) {
        self.state.pendingAttachments.push({
          name: file.name,
          type: file.type,
          size: file.size,
          dataUrl: e.target.result,
        });
        self.renderAttachmentPreview();
      };
      reader.readAsDataURL(file);
    });
    self.toast(files.length + ' image' + (files.length > 1 ? 's' : '') + ' attached', 'success');
  },

  renderAttachmentPreview() {
    var wrapper = document.querySelector('.chat-input-wrapper');
    if (!wrapper) return;
    var existing = document.getElementById('chat-attachments');
    var attachments = this.state.pendingAttachments || [];
    if (attachments.length === 0) {
      if (existing) existing.remove();
      return;
    }
    var self = this;
    var html = attachments.map(function (att, i) {
      return '<div class="chat-attachment">' +
        '<img src="' + att.dataUrl + '" alt="' + self.esc(att.name) + '">' +
        '<button class="chat-attachment-remove" data-idx="' + i + '" title="Remove">×</button>' +
        '</div>';
    }).join('');
    if (existing) {
      existing.innerHTML = html;
    } else {
      var div = document.createElement('div');
      div.id = 'chat-attachments';
      div.className = 'chat-attachments';
      div.innerHTML = html;
      wrapper.insertBefore(div, wrapper.firstChild);
    }
    document.querySelectorAll('.chat-attachment-remove').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.state.pendingAttachments.splice(parseInt(btn.dataset.idx), 1);
        self.renderAttachmentPreview();
      });
    });
  },

  handleSendClick() {
    if (this.state.streaming) this.stopGeneration();
    else this.sendMessage();
  },

  stopGeneration() {
    if (this.state.abortController) {
      this.state.abortController.abort();
      this.state.abortController = null;
    }
    this.state.streaming = false;
    var sendBtn = document.getElementById('chat-send');
    if (sendBtn) { sendBtn.textContent = 'SEND'; sendBtn.classList.remove('streaming'); }
    this.toast('Generation stopped', 'info');
  },

  // ========================================================================
  //  CHAT — MODEL SELECT (Bottom Bar)
  // ========================================================================

  updateChatModelSelect() {
    var select = document.getElementById('chat-model');
    if (!select) return;
    var s = this.state.status;
    if (!s) return;
    // Prefer the fleet-wide union (every node's model) over local models_loaded.
    var models = (this.state.fleetModels && this.state.fleetModels.length)
      ? this.state.fleetModels : (s.models_loaded || []);
    var cv = select.value;
    if (models.length === 0) {
      select.innerHTML = '<option>No models loaded</option>';
    } else {
      var self = this;
      select.innerHTML = models.map(function (m) {
        return '<option value="' + self.esc(m) + '">' + self.esc(m) + '</option>';
      }).join('');
      if (cv) {
        for (var i = 0; i < select.options.length; i++) {
          if (select.options[i].value === cv) { select.value = cv; break; }
        }
      }
    }
  },

  // ========================================================================
  //  CHAT — SEND / STREAM
  // ========================================================================

  async sendMessage() {
    var input = document.getElementById('chat-input');
    var content = input ? input.value.trim() : '';
    if (!content || this.state.streaming) return;
    input.value = '';
    input.style.height = 'auto';

    // Auto-create conversation if needed
    if (!this.state.currentConversation) this.newConversation();

    this.state.messages.push({ role: 'user', content: content });

    // Auto-navigate to chat view when sending
    this.navigate('chat');
    this.renderChatMessages();
    this.showChatOverlay();

    var select = document.getElementById('chat-model');
    var model = select ? select.value : '';

    this.state.streaming = true;
    var sendBtn = document.getElementById('chat-send');
    if (sendBtn) { sendBtn.textContent = 'STOP'; sendBtn.classList.add('streaming'); }

    this.state.streamMetrics = { ttft: null, tps: 0, tokens: 0, startTime: performance.now(), firstTokenTime: 0 };
    this.updateStreamMetrics();

    var assistantMsg = { role: 'assistant', content: '' };
    this.state.messages.push(assistantMsg);
    this.state.abortController = new AbortController();

    try {
      var resp = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: model,
          messages: this.state.messages.slice(0, -1).map(function (m) { return { role: m.role, content: m.content }; }),
          stream: true,
        }),
        signal: this.state.abortController.signal,
      });

      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';

      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        var lines = buffer.split('\n');
        buffer = lines.pop();

        for (var li = 0; li < lines.length; li++) {
          var line = lines[li];
          if (!line.startsWith('data: ')) continue;
          var data = line.slice(6);
          if (data === '[DONE]') break;
          try {
            var json = JSON.parse(data);
            var delta = json.choices && json.choices[0] && json.choices[0].delta && json.choices[0].delta.content;
            if (delta) {
              if (this.state.streamMetrics.tokens === 0) {
                this.state.streamMetrics.firstTokenTime = performance.now();
                this.state.streamMetrics.ttft = Math.round(this.state.streamMetrics.firstTokenTime - this.state.streamMetrics.startTime);
              }
              this.state.streamMetrics.tokens++;
              var elapsed = (performance.now() - this.state.streamMetrics.firstTokenTime) / 1000;
              if (elapsed > 0) this.state.streamMetrics.tps = this.state.streamMetrics.tokens / elapsed;
              assistantMsg.content += delta;
              this.updateStreamingMessage(assistantMsg);
              this.updateStreamMetrics();
            }
          } catch (e) { /* skip parse errors */ }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        if (!assistantMsg.content) this.state.messages.pop();
      } else {
        assistantMsg.content = 'Error: ' + err.message + '. Is the engine running?';
        this.toast('Engine not responding', 'error');
      }
    }

    this.state.streaming = false;
    this.state.abortController = null;
    if (sendBtn) { sendBtn.textContent = 'SEND'; sendBtn.classList.remove('streaming'); }
    this.renderChatMessages();
    this.saveCurrentConversation();
  },

  // ========================================================================
  //  CHAT — MESSAGE RENDERING (center-stage overlay)
  // ========================================================================

  showChatOverlay() {
    var mount = document.getElementById('view-chat-mount');
    if (!mount) return;
    var overlay = document.getElementById('chat-overlay');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'chat-overlay';
      overlay.className = 'chat-overlay';
      overlay.innerHTML = '<div class="chat-metrics-bar" id="chat-metrics-bar">' +
        '<span id="chat-metric-ttft">TTFT: --</span>' +
        '<span id="chat-metric-tps">-- tok/s</span>' +
        '<span id="chat-metric-tokens">0 tokens</span>' +
        '</div>' +
        '<div class="chat-messages" id="chat-messages"></div>';
      mount.appendChild(overlay);
    }
    // Hide the empty state when we have messages
    var empty = document.getElementById('chat-view-empty');
    if (empty) empty.style.display = this.state.messages.length > 0 ? 'none' : '';
    overlay.style.display = this.state.messages.length > 0 ? '' : 'none';
  },

  hideChatOverlay() {
    var overlay = document.getElementById('chat-overlay');
    if (overlay) overlay.style.display = 'none';
    var empty = document.getElementById('chat-view-empty');
    if (empty) empty.style.display = '';
  },

  updateStreamMetrics() {
    var m = this.state.streamMetrics;
    var t1 = document.getElementById('chat-metric-ttft');
    var t2 = document.getElementById('chat-metric-tps');
    var t3 = document.getElementById('chat-metric-tokens');
    if (t1) t1.textContent = m.ttft != null ? 'TTFT: ' + m.ttft + 'ms' : 'TTFT: --';
    if (t2) t2.textContent = m.tps > 0 ? m.tps.toFixed(1) + ' tok/s' : '-- tok/s';
    if (t3) t3.textContent = m.tokens + ' tokens';
  },

  // Targeted update for streaming — only rewrites the current assistant message
  updateStreamingMessage(assistantMsg) {
    var container = document.getElementById('chat-messages');
    if (!container) { this.renderChatMessages(); return; }
    var last = container.lastElementChild;
    if (!last || !last.classList.contains('assistant')) {
      // Not yet rendered — full render once
      this.renderChatMessages();
      return;
    }
    var contentEl = last.querySelector('.chat-msg-content');
    if (!contentEl) { this.renderChatMessages(); return; }
    contentEl.innerHTML = this.formatMarkdown(assistantMsg.content);
    // Auto-scroll to bottom only if user hasn't scrolled up
    var atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 120;
    if (atBottom) container.scrollTop = container.scrollHeight;
  },

  renderChatMessages() {
    // Ensure overlay exists
    this.showChatOverlay();
    var container = document.getElementById('chat-messages');
    if (!container) return;
    var self = this;

    if (this.state.messages.length === 0) {
      this.hideChatOverlay();
      return;
    }

    container.innerHTML = this.state.messages.map(function (msg, i) {
      var cls = msg.role === 'user' ? 'chat-msg user' : 'chat-msg assistant';
      var html = '<div class="' + cls + '">' +
        '<div class="chat-msg-content">' + self.formatMarkdown(msg.content) + '</div>';
      if (msg.role === 'assistant' && msg.content) {
        html += '<button class="chat-copy-btn" data-msg-index="' + i + '" title="Copy">' +
          '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">' +
          '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>' +
          '<path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>' +
          '</svg></button>';
      }
      html += '</div>';
      return html;
    }).join('');

    container.scrollTop = container.scrollHeight;

    container.querySelectorAll('.chat-copy-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var msg = self.state.messages[parseInt(btn.dataset.msgIndex)];
        if (msg) {
          navigator.clipboard.writeText(msg.content).then(function () {
            self.toast('Copied to clipboard', 'success');
          }).catch(function () {
            self.toast('Failed to copy', 'error');
          });
        }
      });
    });

    // Code block copy buttons
    container.querySelectorAll('.code-copy-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var codeEl = document.getElementById(btn.dataset.codeId);
        if (!codeEl) return;
        var code = codeEl.textContent;
        navigator.clipboard.writeText(code).then(function () {
          var label = btn.querySelector('span');
          if (label) {
            var orig = label.textContent;
            label.textContent = 'Copied!';
            btn.classList.add('copied');
            setTimeout(function () {
              label.textContent = orig;
              btn.classList.remove('copied');
            }, 1500);
          }
          self.toast('Code copied', 'success');
        }).catch(function () { self.toast('Failed to copy', 'error'); });
      });
    });

    // Show metrics bar during streaming
    var metricsBar = document.getElementById('chat-metrics-bar');
    if (metricsBar) metricsBar.style.display = this.state.streaming ? '' : 'none';
  },
});
