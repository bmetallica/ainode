/* AINode Command Center — the Config view.
 *
 * Methods of the AINode object, split out of app.js by view (W2 in
 * upgrade-fixes.md) so that work on one view stops conflicting with work
 * on every other. A classic script loaded after app.js: the methods are
 * the same, called the same way, with the same `this`.
 */
Object.assign(AINode, {
  // ========================================================================
  //  CONFIG VIEW
  // ========================================================================

  renderConfig() {
    var nav = document.getElementById('config-nav');
    var self = this;
    if (nav && !nav.dataset.bound) {
      nav.dataset.bound = '1';
      nav.querySelectorAll('.config-nav-item').forEach(function (btn) {
        btn.addEventListener('click', function () {
          self.state.configSection = btn.dataset.section;
          nav.querySelectorAll('.config-nav-item').forEach(function (b) {
            b.classList.toggle('active', b.dataset.section === self.state.configSection);
          });
          self._configViewInitialized = true;
          self.renderConfigSection();
        });
      });
    }
    // Keep sidebar active in sync
    if (nav) {
      nav.querySelectorAll('.config-nav-item').forEach(function (b) {
        b.classList.toggle('active', b.dataset.section === self.state.configSection);
      });
    }
    this.renderConfigSection();
  },

  renderConfigSection() {
    switch (this.state.configSection) {
      case 'credentials': return this.renderConfigCredentials();
      case 'cluster':     return this.renderConfigCluster();
      case 'node':        return this.renderConfigNode();
      case 'storage':     return this.renderConfigStorage();
      case 'memory':      return this.renderConfigMemory();
      case 'updates':     return this.renderConfigUpdates();
      case 'training':    return this.renderConfigTrainingDefaults();
      case 'security':    return this.renderConfigSecurity();
      case 'network':     return this.renderConfigNetwork();
      case 'monitoring':  return this.renderConfigMonitoring();
      case 'about':       return this.renderConfigAbout();
    }
  },

  _configMount() {
    return document.getElementById('config-content');
  },

  async _fetchConfigBundle() {
    var self = this;
    var results = await Promise.all([
      this.fetchJSON('/api/secrets'),
      this.fetchJSON('/api/cluster/info'),
      this.fetchJSON('/api/config'),
    ]);
    self.state.configData.secrets = results[0];
    self.state.configData.cluster = results[1];
    self.state.configData.config = results[2];
  },

  // ----- Security -----------------------------------------------------------

  async renderConfigSecurity() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading security settings…</div>';
    var data = await this.fetchJSON('/api/auth/status');
    if (!data) {
      mount.innerHTML = '<div class="config-empty">Unable to load auth status.</div>';
      return;
    }
    var self = this;
    var on = !!data.enabled;
    var keys = data.keys || [];

    var web = (await this.fetchJSON('/api/auth/web/status')) || {};

    var html = '';
    html += '<h2 class="config-section-title">Security</h2>';
    html += '<p class="config-section-desc">The web UI and every <span class="mono">/api/</span> route are behind a ' +
            'sign-in. The other nodes get past it with a shared cluster key, and this node\'s own scripts from ' +
            'localhost. The OpenAI-compatible endpoint <span class="mono">/v1/</span> is separate: clients such as ' +
            'Open WebUI cannot sign in to a web page, so it is guarded by the optional API keys below.</p>';

    html += '<div class="config-card">';
    html += '<h3 class="config-card-title">Web sign-in</h3>';
    if (web.is_default) {
      html += '<div class="config-warning">This node still uses the default password. Anyone on the ' +
              'network who guesses <span class="mono">admin / admin</span> can control it — change it below.</div>';
    }
    html += '<div class="config-row"><div>Signed in as <strong class="mono">' +
            self.esc(web.user || '') + '</strong></div>' +
            '<button class="config-btn secondary" id="cfg-web-logout">Sign out</button></div>';
    html += '<div class="config-password-form">';
    html += '  <label class="form-label" for="cfg-web-user">User name</label>';
    html += '  <input class="form-input" id="cfg-web-user" autocomplete="username" value="' + self.esc(web.user || 'admin') + '">';
    html += '  <label class="form-label" for="cfg-web-current">Current password</label>';
    html += '  <input class="form-input" id="cfg-web-current" type="password" autocomplete="current-password">';
    html += '  <label class="form-label" for="cfg-web-new">New password</label>';
    html += '  <input class="form-input" id="cfg-web-new" type="password" autocomplete="new-password" placeholder="at least 8 characters">';
    html += '  <label class="form-label" for="cfg-web-repeat">Repeat new password</label>';
    html += '  <input class="form-input" id="cfg-web-repeat" type="password" autocomplete="new-password">';
    html += '  <label class="config-check"><input type="checkbox" id="cfg-web-all" checked> Set it on every node of the cluster</label>';
    html += '  <button class="config-btn primary" id="cfg-web-save">Change password</button>';
    html += '  <div class="config-field-hint" id="cfg-web-result"></div>';
    html += '</div>';
    html += '<p class="config-card-desc">Changing the password signs every other browser out. Forgotten it? ' +
            'On the node: <span class="mono">rm ~/.ainode/web-auth.json</span> and restart AINode — it comes back as ' +
            '<span class="mono">admin / admin</span>.</p>';
    html += '</div>';

    html += '<div class="config-card">';
    html += '<h3 class="config-card-title">API keys for /v1/</h3>';
    html += '<p class="config-card-desc">Off by default. When on, OpenAI-compatible clients must send a key; ' +
            'the dashboard itself does not need one, because it is signed in.</p>';
    html += '<div class="config-row">';
    html += '  <div><span class="config-auth-state ' + (on ? 'on' : 'off') + '">' +
            (on ? 'ENABLED' : 'DISABLED') + '</span>' +
            '<span class="config-auth-count">' + keys.length + ' key' + (keys.length === 1 ? '' : 's') + '</span></div>';
    html += '  <button class="config-btn' + (on ? ' danger' : '') + '" id="cfg-auth-toggle">' +
            (on ? 'Disable' : 'Enable') + '</button>';
    html += '</div>';
    if (!on) {
      html += '<p class="config-card-desc">Enabling generates a first key and shows it once.</p>';
    }
    html += '</div>';

    if (on) {
      html += '<div class="config-card">';
      html += '<h3 class="config-card-title">API keys</h3>';
      html += '<p class="config-card-desc">A key is shown once, at creation — only its hash is stored, so it cannot be recovered. Give each person their own so one can be revoked without disturbing the others.</p>';
      if (!keys.length) {
        html += '<div class="config-empty">No keys. Create one below, or nobody can reach the API.</div>';
      } else {
        keys.forEach(function (k) {
          var when = k.created_at ? new Date(k.created_at * 1000).toLocaleString() : '';
          html += '<div class="config-secret-row">';
          html += '  <div class="config-secret-main"><div class="config-secret-label-row">';
          html += '    <span class="config-secret-name">' + self.esc(k.name || '(unnamed)') + '</span>';
          html += '    <span class="config-secret-mask mono">' + self.esc(k.id || '') + '</span>';
          html += '  </div>' + (when ? '<div class="config-key-when">created ' + self.esc(when) + '</div>' : '') + '</div>';
          html += '  <span></span>';
          html += '  <button class="config-icon-btn danger" data-revoke="' + self.esc(k.id || '') + '">Revoke</button>';
          html += '</div>';
        });
      }
      html += '<div class="config-add-custom">';
      html += '  <input class="form-input" id="cfg-key-name" placeholder="label, e.g. anna\'s laptop">';
      html += '  <button class="config-btn" id="cfg-key-add">+ Create key</button>';
      html += '</div>';
      html += '</div>';

      html += '<div class="config-card">';
      html += '<h3 class="config-card-title">Using a key</h3>';
      html += '<p class="config-card-desc">Any OpenAI-compatible client works — point it at this node and pass the key as the API key.</p>';
      // Port 3000, the port AINode listens on — not 8000, which is one engine's
      // own listener and knows nothing of the keys, the router or the fleet.
      html += '<pre class="config-code mono">curl ' + self.esc(location.origin) +
              '/v1/chat/completions \\\n  -H "Authorization: Bearer &lt;your-key&gt;" \\\n' +
              '  -H "Content-Type: application/json" \\\n  -d \'{"model":"...","messages":[...]}\'</pre>';
      html += '</div>';
    }

    mount.innerHTML = html;

    var logout = document.getElementById('cfg-web-logout');
    if (logout) logout.addEventListener('click', function () { self.signOut(); });

    var save = document.getElementById('cfg-web-save');
    if (save) {
      save.addEventListener('click', async function () {
        var value = function (id) { var el = document.getElementById(id); return el ? el.value : ''; };
        var result = document.getElementById('cfg-web-result');
        if (value('cfg-web-new') !== value('cfg-web-repeat')) {
          result.textContent = 'The two new passwords do not match.';
          return;
        }
        save.disabled = true;
        var resp = await fetch('/api/auth/web/password', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user: value('cfg-web-user'), current: value('cfg-web-current'),
            new: value('cfg-web-new'),
            all_nodes: !!(document.getElementById('cfg-web-all') || {}).checked,
          }),
        });
        var body = await resp.json().catch(function () { return {}; });
        save.disabled = false;
        if (!resp.ok) {
          result.textContent = body.error || 'Could not change the password.';
          return;
        }
        var nodes = body.nodes || {};
        var failed = Object.keys(nodes).filter(function (k) { return nodes[k] !== 'ok'; });
        self.toast('Password changed' + (Object.keys(nodes).length
          ? ' on this node and ' + (Object.keys(nodes).length - failed.length) + ' other(s)' : ''),
          failed.length ? 'info' : 'success');
        if (failed.length) {
          result.textContent = 'Not set on: ' + failed.map(function (k) {
            return k + ' (' + nodes[k] + ')';
          }).join(', ') + '. Those keep their old password until they are reachable and it is set again.';
        }
        self.state.webDefaultPassword = false;
        self.renderDefaultPasswordBanner();
        self.renderConfigSecurity();
      });
    }

    var toggle = document.getElementById('cfg-auth-toggle');
    if (toggle) {
      toggle.addEventListener('click', async function () {
        if (on) {
          if (!confirm('Turn off API keys? /v1/ becomes reachable without a key. The web sign-in stays.')) return;
          await fetch('/api/auth/disable', { method: 'POST' });
          self.toast('Authentication disabled', 'info');
          self.renderConfigSecurity();
          return;
        }
        var resp = await fetch('/api/auth/enable', { method: 'POST' });
        var body = await resp.json().catch(function () { return {}; });
        if (body.api_key) self._showOneTimeKey(body.api_key);
        self.toast('Authentication enabled', 'success');
        self.renderConfigSecurity();
      });
    }

    var addKey = document.getElementById('cfg-key-add');
    if (addKey) {
      addKey.addEventListener('click', async function () {
        var nameEl = document.getElementById('cfg-key-name');
        var resp = await fetch('/api/auth/keys', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: nameEl ? nameEl.value : '' }),
        });
        var body = await resp.json().catch(function () { return {}; });
        if (body.api_key) self._showOneTimeKey(body.api_key);
        self.renderConfigSecurity();
      });
    }

    mount.querySelectorAll('[data-revoke]').forEach(function (btn) {
      btn.addEventListener('click', async function () {
        var id = btn.dataset.revoke;
        if (!confirm('Revoke key ' + id + '? Any client using it stops working immediately.')) return;
        await fetch('/api/auth/keys/' + encodeURIComponent(id), { method: 'DELETE' });
        self.toast('Revoked ' + id, 'info');
        self.renderConfigSecurity();
      });
    });
  },

  // A key exists in plaintext exactly once, in this response. Show it in a
  // blocking panel with a copy button rather than a toast that scrolls away.
  _showOneTimeKey(key) {
    var self = this;
    var wrap = document.createElement('div');
    wrap.className = 'onetime-key-backdrop';
    wrap.innerHTML =
      '<div class="onetime-key">' +
      '  <h3>Your new API key</h3>' +
      '  <p>Copy it now — it is stored only as a hash and cannot be shown again.</p>' +
      '  <div class="onetime-key-value mono" id="onetime-key-value">' + self.esc(key) + '</div>' +
      '  <div class="onetime-key-actions">' +
      '    <button class="config-btn" id="onetime-key-copy">Copy</button>' +
      '    <button class="config-btn" id="onetime-key-done">Done</button>' +
      '  </div>' +
      '</div>';
    document.body.appendChild(wrap);
    var copy = wrap.querySelector('#onetime-key-copy');
    copy.addEventListener('click', function () {
      if (navigator.clipboard) {
        navigator.clipboard.writeText(key).then(function () { copy.textContent = 'Copied'; });
      } else {
        // Clipboard API needs a secure context; a LAN node on plain http has none.
        var v = document.getElementById('onetime-key-value');
        var r = document.createRange();
        r.selectNodeContents(v);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(r);
        copy.textContent = 'Selected — press Ctrl+C';
      }
    });
    wrap.querySelector('#onetime-key-done').addEventListener('click', function () {
      wrap.remove();
    });
  },

  // ----- Credentials --------------------------------------------------------
  async renderConfigCredentials() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading credentials…</div>';
    var data = await this.fetchJSON('/api/secrets');
    this.state.configData.secrets = data;
    if (!data) {
      mount.innerHTML = '<div class="config-empty">Unable to load secrets.</div>';
      return;
    }
    var self = this;
    var known = data.known || {};
    var custom = data.custom || {};
    var html = '';
    html += '<h2 class="config-section-title">Credentials</h2>';
    html += '<p class="config-section-desc">Store API tokens used by AINode to download gated models, log training metrics, and access hosted services. Values are stored locally at <code>~/.ainode/secrets.json</code> with mode 0600 and never transmitted to AINode servers.</p>';

    html += '<div class="config-card">';
    html += '<h3 class="config-card-title">Known credentials</h3>';
    html += '<p class="config-card-desc">Recognized services with built-in integrations.</p>';
    Object.keys(known).forEach(function (k) {
      html += self._renderSecretRow(known[k]);
    });
    html += '</div>';

    html += '<div class="config-card">';
    html += '<h3 class="config-card-title">Custom secrets</h3>';
    html += '<p class="config-card-desc">Arbitrary named values, e.g. for training scripts or third-party integrations.</p>';
    if (Object.keys(custom).length === 0) {
      html += '<div class="config-empty">No custom secrets yet.</div>';
    } else {
      Object.keys(custom).forEach(function (name) {
        var s = custom[name];
        html += '<div class="config-secret-row">';
        html += '  <div class="config-secret-main">';
        html += '    <div class="config-secret-label-row">';
        html += '      <span class="config-secret-name">' + self.esc(name) + '</span>';
        html += s.is_set ? '      <span class="config-secret-mask">' + self.esc(s.masked) + '</span>' : '      <span class="config-secret-unset">not set</span>';
        html += '    </div>';
        html += '  </div>';
        html += '  <span></span>';
        html += '  <button class="config-icon-btn danger" data-custom-delete="' + self.esc(name) + '">Delete</button>';
        html += '</div>';
      });
    }
    html += '<div class="config-add-custom">';
    html += '  <input class="form-input" id="cfg-custom-name" placeholder="name (alphanumeric)">';
    html += '  <input class="form-input" id="cfg-custom-value" type="password" placeholder="value">';
    html += '  <button class="config-btn" id="cfg-custom-add">+ Add</button>';
    html += '</div>';
    html += '</div>';

    mount.innerHTML = html;

    // Wire save buttons for each known secret
    Object.keys(known).forEach(function (k) {
      self._wireSecretRow(k);
    });

    mount.querySelectorAll('[data-custom-delete]').forEach(function (btn) {
      btn.addEventListener('click', async function () {
        var name = btn.dataset.customDelete;
        if (!confirm('Delete custom secret "' + name + '"?')) return;
        await fetch('/api/secrets/custom/' + encodeURIComponent(name), { method: 'DELETE' });
        self.toast('Deleted ' + name, 'info');
        self.renderConfigCredentials();
      });
    });

    var addBtn = document.getElementById('cfg-custom-add');
    if (addBtn) {
      addBtn.addEventListener('click', async function () {
        var nameEl = document.getElementById('cfg-custom-name');
        var valEl = document.getElementById('cfg-custom-value');
        var name = (nameEl.value || '').trim();
        var value = valEl.value || '';
        if (!name || !value) { self.toast('Name and value required', 'error'); return; }
        var resp = await fetch('/api/secrets/custom/' + encodeURIComponent(name), {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ value: value }),
        });
        var body = await resp.json().catch(function () { return {}; });
        if (!resp.ok) {
          self.toast((body.error && body.error.message) || 'Failed to add secret', 'error');
          return;
        }
        self.toast('Added ' + name, 'success');
        self.renderConfigCredentials();
      });
    }
  },

  _renderSecretRow(entry) {
    var self = this;
    var k = entry.key;
    var html = '<div class="config-secret-row" data-secret-row="' + self.esc(k) + '">';
    html += '  <div class="config-secret-main">';
    html += '    <div class="config-secret-label-row">';
    html += '      <span class="config-secret-name">' + self.esc(entry.label) + '</span>';
    html += entry.is_set
      ? '      <span class="config-secret-mask">' + self.esc(entry.masked) + '</span>'
      : '      <span class="config-secret-unset">not set</span>';
    html += '    </div>';
    html += '    <div class="config-secret-desc">' + self.esc(entry.description || '') +
            (entry.docs_url ? ' <a href="' + self.esc(entry.docs_url) + '" target="_blank" rel="noopener" style="color:var(--nvidia-green)">Docs ↗</a>' : '') +
            '</div>';
    html += '    <div class="config-secret-input-row" id="cfg-sec-input-' + self.esc(k) + '" style="' + (entry.is_set ? 'display:none' : '') + '">';
    html += '      <input class="form-input" type="password" id="cfg-sec-val-' + self.esc(k) + '" placeholder="' + self.esc(entry.prefix_hint ? entry.prefix_hint + '…' : 'paste token here') + '">';
    html += '      <button class="config-eye-btn" data-eye="' + self.esc(k) + '" type="button">Show</button>';
    html += '      <button class="config-btn" data-save-secret="' + self.esc(k) + '">Save</button>';
    html += '    </div>';
    html += '    <div class="config-test-result" id="cfg-sec-result-' + self.esc(k) + '" style="display:none"></div>';
    html += '  </div>';
    if (entry.is_set) {
      html += '  <button class="config-btn secondary" data-replace-secret="' + self.esc(k) + '">Replace</button>';
      html += '  <button class="config-btn danger" data-delete-secret="' + self.esc(k) + '">Delete</button>';
    } else {
      html += '  <span></span><span></span>';
    }
    if (entry.testable) {
      html += '<div style="grid-column: 1 / -1; margin-top: 6px; text-align: right;"><button class="config-btn secondary" data-test-secret="' + self.esc(k) + '"' + (entry.is_set ? '' : ' disabled') + '>Test connection</button></div>';
    }
    html += '</div>';
    return html;
  },

  _wireSecretRow(key) {
    var self = this;
    var mount = this._configMount();
    if (!mount) return;
    var row = mount.querySelector('[data-secret-row="' + key + '"]');
    if (!row) return;

    var eye = row.querySelector('[data-eye="' + key + '"]');
    if (eye) eye.addEventListener('click', function () {
      var inp = document.getElementById('cfg-sec-val-' + key);
      if (!inp) return;
      var shown = inp.type === 'text';
      inp.type = shown ? 'password' : 'text';
      eye.textContent = shown ? 'Show' : 'Hide';
    });

    var save = row.querySelector('[data-save-secret="' + key + '"]');
    if (save) save.addEventListener('click', async function () {
      var inp = document.getElementById('cfg-sec-val-' + key);
      var val = inp ? inp.value : '';
      if (!val) { self.toast('Value is required', 'error'); return; }
      var resp = await fetch('/api/secrets/' + encodeURIComponent(key), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value: val }),
      });
      var body = await resp.json().catch(function () { return {}; });
      if (!resp.ok) {
        self.toast((body.error && body.error.message) || 'Save failed', 'error');
        return;
      }
      if (inp) inp.value = '';
      self.toast('Saved', 'success');
      self.renderConfigCredentials();
    });

    var replace = row.querySelector('[data-replace-secret="' + key + '"]');
    if (replace) replace.addEventListener('click', function () {
      var wrap = document.getElementById('cfg-sec-input-' + key);
      if (wrap) wrap.style.display = '';
      replace.style.display = 'none';
    });

    var del = row.querySelector('[data-delete-secret="' + key + '"]');
    if (del) del.addEventListener('click', async function () {
      if (!confirm('Delete ' + key + '?')) return;
      await fetch('/api/secrets/' + encodeURIComponent(key), { method: 'DELETE' });
      self.toast('Deleted', 'info');
      self.renderConfigCredentials();
    });

    var test = row.querySelector('[data-test-secret="' + key + '"]');
    if (test) test.addEventListener('click', async function () {
      var result = document.getElementById('cfg-sec-result-' + key);
      if (result) { result.style.display = ''; result.className = 'config-test-result'; result.textContent = 'Testing…'; }
      var resp = await fetch('/api/secrets/' + encodeURIComponent(key) + '/test');
      var body = await resp.json().catch(function () { return {}; });
      if (!result) return;
      if (body.ok) {
        result.className = 'config-test-result ok';
        result.textContent = 'OK — authenticated as ' + (body.identity || 'user');
      } else {
        result.className = 'config-test-result err';
        result.textContent = 'Failed: ' + (body.message || 'unknown error');
      }
    });
  },

  // ----- Cluster ------------------------------------------------------------
  async renderConfigCluster() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading cluster info…</div>';
    var data = await this.fetchJSON('/api/cluster/info');
    this.state.configData.cluster = data;
    if (!data) {
      mount.innerHTML = '<div class="config-empty">Unable to load cluster info.</div>';
      return;
    }
    var self = this;
    var role = data.my_role || 'worker';
    var configured = data.configured_role || 'auto';
    var masterId = data.master_node_id;
    var iAmMaster = role === 'master';
    var members = data.members || [];
    var peers = members.filter(function (m) { return m.node_id !== data.my_node_id; });

    var badgeTitle, badgeSub, badgeIcon, badgeCls;
    if (iAmMaster) {
      badgeIcon = '⚡';
      badgeTitle = 'MASTER — You are the cluster head';
      badgeSub = peers.length > 0 ? 'Serving ' + peers.length + ' worker(s)'
                                   : (configured === 'auto' ? 'Waiting for peers (auto-elected)' : 'Standalone');
      badgeCls = 'master';
    } else {
      badgeIcon = '🔗';
      var masterNode = members.find(function (m) { return m.node_id === masterId; });
      badgeTitle = 'WORKER';
      badgeSub = masterNode
        ? 'Connected to master: ' + masterNode.node_name + ' (' + (data.master_address || masterNode.node_id) + ')'
        : 'No master detected yet';
      badgeCls = 'worker';
    }

    var html = '';
    html += '<h2 class="config-section-title">Cluster</h2>';
    html += '<p class="config-section-desc">Nodes on the same network with a matching <code>cluster_id</code> form a cluster. Master is elected automatically, or you can pin a role explicitly.</p>';

    html += '<div class="config-role-badge ' + badgeCls + '">';
    html += '  <div class="config-role-badge-icon">' + badgeIcon + '</div>';
    html += '  <div class="config-role-badge-text">';
    html += '    <div class="config-role-badge-title">' + self.esc(badgeTitle) + '</div>';
    html += '    <div class="config-role-badge-subtitle">' + self.esc(badgeSub) + '</div>';
    html += '  </div>';
    html += '</div>';

    html += '<div class="config-card">';
    html += '  <h3 class="config-card-title">Role</h3>';
    html += '  <p class="config-card-desc">Auto = elected dynamically · Master = always cluster head · Worker = never becomes master.</p>';
    html += '  <div class="config-role-pills pill-group">';
    ['auto', 'master', 'worker'].forEach(function (r) {
      html += '<button class="pill' + (configured === r ? ' active' : '') + '" data-set-role="' + r + '">' + r.toUpperCase() + '</button>';
    });
    html += '  </div>';
    html += '</div>';

    html += '<div class="config-card">';
    html += '  <h3 class="config-card-title">Cluster ID</h3>';
    html += '  <p class="config-card-desc">Only nodes sharing this identifier will see each other.</p>';
    html += '  <div class="config-secret-input-row">';
    html += '    <input class="form-input" id="cfg-cluster-id" value="' + self.esc(data.cluster_id || 'default') + '">';
    html += '    <button class="config-btn" id="cfg-cluster-id-save">Save</button>';
    html += '  </div>';
    html += '</div>';

    html += '<div class="config-card">';
    html += '  <h3 class="config-card-title">Members</h3>';
    html += '  <p class="config-card-desc">Nodes currently visible in this cluster.</p>';
    if (members.length === 0) {
      html += '<div class="config-empty">No members yet.</div>';
    } else {
      html += '<div class="config-member-table">';
      html += '<div class="config-member-row header"><span>Node</span><span>Address</span><span>Role</span><span>Status</span><span>Last seen</span></div>';
      members.forEach(function (m) {
        var when = m.last_seen ? new Date(m.last_seen * 1000).toLocaleTimeString() : '—';
        html += '<div class="config-member-row">';
        html += '  <span><strong>' + self.esc(m.node_name || m.node_id) + '</strong><div style="color:var(--text-muted);font-size:11px" class="mono">' + self.esc(m.node_id) + '</div></span>';
        html += '  <span class="mono">' + self.esc(m.node_name) + ':' + self.esc(String(m.web_port)) + '</span>';
        html += '  <span><span class="config-member-role-tag ' + m.effective_role + '">' + m.effective_role + '</span></span>';
        html += '  <span>' + self.esc(m.status) + '</span>';
        html += '  <span class="mono" style="color:var(--text-muted)">' + when + '</span>';
        html += '</div>';
      });
      html += '</div>';
    }
    html += '</div>';

    html += '<div class="config-card">';
    html += '  <h3 class="config-card-title">How election works</h3>';
    html += '  <p class="config-card-desc">If any node is explicitly configured as <code>master</code>, the lowest-ID such node wins. Otherwise the lowest node_id among <code>auto</code> nodes becomes master. <code>worker</code> nodes are never elected. Election is re-run every 5s as heartbeats arrive; changing your role takes effect on the next broadcast (≤5s) without a restart.</p>';
    html += '</div>';

    mount.innerHTML = html;

    mount.querySelectorAll('[data-set-role]').forEach(function (btn) {
      btn.addEventListener('click', async function () {
        var role = btn.dataset.setRole;
        var resp = await fetch('/api/cluster/role', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ role: role }),
        });
        if (!resp.ok) { self.toast('Role change failed', 'error'); return; }
        self.toast('Role → ' + role.toUpperCase(), 'success');
        setTimeout(function () { self.renderConfigCluster(); }, 500);
      });
    });

    var idBtn = document.getElementById('cfg-cluster-id-save');
    if (idBtn) idBtn.addEventListener('click', async function () {
      var val = document.getElementById('cfg-cluster-id').value.trim() || 'default';
      var resp = await fetch('/api/cluster/id', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cluster_id: val }),
      });
      var body = await resp.json().catch(function () { return {}; });
      if (!resp.ok) { self.toast((body.error && body.error.message) || 'Save failed', 'error'); return; }
      self.toast('Cluster ID saved', 'success');
      setTimeout(function () { self.renderConfigCluster(); }, 500);
    });
  },

  // ----- Node Identity ------------------------------------------------------
  async renderConfigNode() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading…</div>';
    var cfg = await this.fetchJSON('/api/config');
    this.state.configData.config = cfg;
    if (!cfg) { mount.innerHTML = '<div class="config-empty">Unable to load config.</div>'; return; }
    var self = this;
    var html = '';
    html += '<h2 class="config-section-title">Node Identity</h2>';
    html += '<p class="config-section-desc">Human-readable name and owner contact for this node.</p>';
    html += '<div class="config-card"><div class="config-form-grid">';
    html += this._field('Node name', 'node_name', cfg.node_name || '');
    html += this._field('Node ID (read-only)', 'node_id', cfg.node_id || '', { readonly: true, mono: true });
    html += this._field('Email', 'email', cfg.email || '', { type: 'email' });
    html += '</div>';
    html += '<div class="config-actions"><button class="config-btn" id="cfg-node-save">Save</button></div>';
    html += '</div>';
    mount.innerHTML = html;
    document.getElementById('cfg-node-save').addEventListener('click', function () {
      self._patchConfig({
        node_name: document.getElementById('cfg-f-node_name').value,
        email: document.getElementById('cfg-f-email').value,
      });
    });
  },

  // ----- Storage ------------------------------------------------------------
  async renderConfigStorage() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading…</div>';
    var cfg = await this.fetchJSON('/api/config');
    this.state.configData.config = cfg;
    if (!cfg) { mount.innerHTML = '<div class="config-empty">Unable to load config.</div>'; return; }
    var self = this;
    var html = '';
    html += '<h2 class="config-section-title">Storage</h2>';
    html += '<p class="config-section-desc">Where AINode keeps downloaded models, datasets, and training artifacts. Leave blank to use defaults under <code>~/.ainode/</code>.</p>';
    html += '<div class="config-card"><div class="config-form-grid single">';
    html += this._field('Models directory', 'models_dir', cfg.models_dir || '', { hint: 'vLLM/HF model weights' });
    html += this._field('Datasets directory', 'datasets_dir', cfg.datasets_dir || '', { hint: 'Training / eval datasets' });
    html += this._field('Training output directory', 'training_dir', cfg.training_dir || '', { hint: 'Checkpoints and logs' });
    html += this._field('HuggingFace cache', 'hf_cache_dir', cfg.hf_cache_dir || '', { hint: 'Defaults to $HF_HOME / ~/.cache/huggingface' });
    html += '</div>';
    html += '<div class="config-actions"><button class="config-btn" id="cfg-storage-save">Save</button></div>';
    html += '</div>';
    mount.innerHTML = html;
    document.getElementById('cfg-storage-save').addEventListener('click', function () {
      self._patchConfig({
        models_dir: document.getElementById('cfg-f-models_dir').value,
        datasets_dir: document.getElementById('cfg-f-datasets_dir').value,
        training_dir: document.getElementById('cfg-f-training_dir').value,
        hf_cache_dir: document.getElementById('cfg-f-hf_cache_dir').value,
      });
    });
  },

  // ----- Memory Guard -------------------------------------------------------
  // Per node, and every node at once, because the reserve is a per-node
  // setting and the nodes that ran out are not the one whose UI is open.
  async renderConfigMemory() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading…</div>';
    var self = this;
    var results = await Promise.all([
      this.fetchJSON('/api/cluster/safety/memory'),
      // What the guard has actually stopped, fleet-wide. The node that ran
      // out is rarely the one you are looking at, and the record it wrote
      // lives there.
      this.fetchJSON('/api/cluster/measurements'),
    ]);
    var data = results[0];
    var rows = (data && data.nodes) || [];
    var blocked = this._blockedModels(results[1]);
    if (!rows.length) {
      mount.innerHTML = '<div class="config-empty">No node reported a memory ' +
        'guard. A node running an older build has none.</div>';
      return;
    }

    var html = '';
    html += '<h2 class="config-section-title">Memory Guard</h2>';
    html += '<p class="config-section-desc">On this hardware the GPU ' +
      'allocation and the operating system share one pool, so an engine that ' +
      'over-allocates does not fail with a CUDA error — it starves the kernel ' +
      'and the node has to be power-cycled. Below the <strong>warning</strong> ' +
      'line no new model may be launched; below the <strong>critical</strong> ' +
      'line the most recently started engine is killed. Set per node — the ' +
      'node that runs out is rarely the one you are looking at.</p>';

    html += '<div class="config-card">';
    html += '<div class="config-actions" style="justify-content:flex-start;gap:8px">';
    html += '<button class="config-btn" data-mem-preset="dgx-spark">DGX Spark preset (8 / 4 GB)</button>';
    html += '<button class="config-btn" data-mem-preset="generic">Generic preset (2 / 1 GB)</button>';
    html += '<span class="config-field-hint" style="align-self:center">applies to every node</span>';
    html += '</div></div>';

    // The planning headroom: one figure for the cluster, because every node
    // that leads a launch checks it with its own copy.
    var local = rows.find(function (r) { return r.reachable && r.available; }) || {};
    var configured = local.plan_headroom_gb;
    html += '<div class="config-card">';
    html += '<h3 class="config-card-title">Planning headroom</h3>';
    html += '<p class="config-card-desc">What the planner and the profile wizard keep ' +
      'free above the <strong>warning</strong> line on every node, so a launch that ' +
      'went exactly to plan still has slack. Not a brake: the guard lines above are ' +
      'the brake, and a model that has already run at its size is checked against ' +
      'them alone. Less headroom is more KV cache — about 26,000 tokens of ' +
      'Qwen3.8-27B per GB. Empty: automatic, 6% of the node (1–8 GB).</p>';
    html += '<div class="config-form-grid"><div><label class="config-field-label">' +
      'Headroom (GB per node)</label><input class="form-input" type="number" min="0" ' +
      'max="30" step="0.5" id="mem-headroom" value="' +
      (configured === null || configured === undefined ? '' : configured) +
      '" placeholder="automatic (' + (local.plan_headroom_effective_gb || '—') +
      ' GB)"></div></div>';
    html += '<p class="config-card-desc">Now: ' + rows.filter(function (r) {
      return r.reachable && r.available;
    }).map(function (r) {
      return self.esc(r.node_name || r.node_id) + ' ' +
        (r.plan_headroom_effective_gb != null ? r.plan_headroom_effective_gb + ' GB' : '—');
    }).join(' · ') + '</p>';
    html += '<div class="config-actions"><button class="config-btn" ' +
      'data-mem-headroom>Save for all nodes</button></div></div>';

    html += this._blockedModelsCard(blocked);

    rows.forEach(function (row) {
      var id = row.node_id;
      var name = row.node_name || id;
      var free = row.available_mb ? (row.available_mb / 1024).toFixed(1) : null;
      var total = row.total_mb ? (row.total_mb / 1024).toFixed(0) : null;
      var enforced = row.warn_mb
        ? (row.warn_mb / 1024).toFixed(1) + ' / ' + (row.critical_mb / 1024).toFixed(1) + ' GB'
        : '—';
      html += '<div class="config-card">';
      html += '<h3 class="config-card-title">' + self.esc(name) + '</h3>';
      if (!row.reachable) {
        html += '<p class="config-card-desc">Not reachable right now.</p></div>';
        return;
      }
      if (!row.available) {
        html += '<p class="config-card-desc">This node runs a build without ' +
          'the memory guard.</p></div>';
        return;
      }
      html += '<p class="config-card-desc">' +
        (free ? free + ' GB free of ' + total + ' GB' : 'memory unreadable') +
        ' · enforcing <strong>' + enforced + '</strong>' +
        (row.blocking ? ' · <span style="color:#ffb84d">refusing new launches</span>' : '') +
        (row.enabled ? '' : ' · <span style="color:#ff5c5c">disabled</span>') +
        '</p>';
      // The configured value and the enforced one differ only when the
      // reserve is larger than a share of the machine's total memory — which
      // never happens on a 128 GB node, and stops an 8 GB one refusing every
      // launch. Both are shown so the difference is never a surprise.
      if (row.warn_gb && Math.abs(row.warn_gb - row.warn_mb / 1024) > 0.05) {
        html += '<p class="config-card-desc">Configured ' + row.warn_gb +
          ' / ' + row.critical_gb + ' GB, capped to the enforced values above ' +
          'because this machine is too small to hold that much back.</p>';
      }
      html += '<div class="config-form-grid">';
      html += '<div><label class="config-field-label">Warning (GB)</label>' +
        '<input class="form-input" type="number" min="0" step="0.5" ' +
        'id="mem-warn-' + self.esc(id) + '" value="' + (row.warn_gb || 0) + '"></div>';
      html += '<div><label class="config-field-label">Critical (GB)</label>' +
        '<input class="form-input" type="number" min="0" step="0.5" ' +
        'id="mem-crit-' + self.esc(id) + '" value="' + (row.critical_gb || 0) + '"></div>';
      html += '</div>';
      html += '<label class="launch-pin" style="margin-top:10px">' +
        '<input type="checkbox" id="mem-on-' + self.esc(id) + '"' +
        (row.enabled ? ' checked' : '') + '> Guard enabled on this node</label>';
      html += '<div class="config-actions">' +
        '<button class="config-btn" data-mem-save="' + self.esc(id) + '">Save</button>' +
        '</div></div>';
    });
    mount.innerHTML = html;

    mount.querySelectorAll('[data-mem-save]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var id = btn.getAttribute('data-mem-save');
        self._saveMemoryGuard({
          node_id: id,
          warn_gb: parseFloat(document.getElementById('mem-warn-' + id).value),
          critical_gb: parseFloat(document.getElementById('mem-crit-' + id).value),
          enabled: document.getElementById('mem-on-' + id).checked,
        });
      });
    });
    mount.querySelectorAll('[data-mem-headroom]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var raw = document.getElementById('mem-headroom').value.trim();
        self._saveMemoryGuard({ all: true, plan_headroom_gb: raw === '' ? null : parseFloat(raw) });
      });
    });
    mount.querySelectorAll('[data-unlock-model]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.unlockModel(btn.getAttribute('data-unlock-model'),
                         btn.getAttribute('data-unlock-node'));
      });
    });
    mount.querySelectorAll('[data-mem-preset]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self._saveMemoryGuard({ all: true,
                                preset: btn.getAttribute('data-mem-preset') });
      });
    });
  },

  // Every model the guard has stopped somewhere, newest first.
  // /api/cluster/measurements answers {models: {name: [entry per node]}} —
  // one row per model PER NODE, because the guard that stopped it is the one
  // on the node it ran on, and that is where the record to clear lives.
  _blockedModels(cluster) {
    var out = [];
    var models = (cluster && cluster.models) || {};
    Object.keys(models).forEach(function (model) {
      (models[model] || []).forEach(function (m) {
        if (!m.guard_stops) return;
        out.push({
          model: model,
          nodeId: m.node_id || '',
          nodeName: m.node_id || 'this node',
          stops: m.guard_stops,
          at: m.last_guard_stop || 0,
          gmu: m.guard_stop_gmu || 0,
          maxLen: m.guard_stop_max_model_len || 0,
          nodes: m.guard_stop_nodes || 0,
          args: m.guard_stop_args || [],
        });
      });
    });
    return out.sort(function (a, b) { return b.at - a.at; });
  },

  _blockedModelsCard(blocked) {
    var self = this;
    var html = '<div class="config-card">';
    html += '<h3 class="config-card-title">Stopped by the guard</h3>';
    if (!blocked.length) {
      html += '<p class="config-card-desc">Nothing is blocked. A model the ' +
        'guard has had to kill is refused the next time it is launched the ' +
        'same way — it would appear here, with a button to lift that.</p>';
      return html + '</div>';
    }
    html += '<p class="config-card-desc">These launches ran the node out of ' +
      'memory and an engine had to be killed. Each is refused until it is ' +
      'launched differently — fewer tokens, a lower utilization, more nodes, ' +
      'or a flag the killed one did not carry. <strong>Unlock</strong> drops ' +
      'that record: use it when the cause has been fixed by something the ' +
      'record cannot see. The measurements are kept.</p>';
    blocked.forEach(function (b) {
      var when = b.at ? new Date(b.at * 1000).toLocaleString() : 'at an unknown time';
      var asked = [];
      if (b.gmu) asked.push('gpu-memory-utilization ' + b.gmu.toFixed(2));
      if (b.maxLen) asked.push('max-model-len ' + b.maxLen.toLocaleString());
      if (b.nodes) asked.push(b.nodes + ' node' + (b.nodes === 1 ? '' : 's'));
      html += '<div class="config-row" style="align-items:flex-start">';
      html += '<div><div><strong>' + self.esc(b.model) + '</strong>' +
        '<span class="config-field-hint"> · ' + self.esc(b.nodeName) + '</span></div>' +
        '<div class="config-field-hint">' + b.stops + ' stop' +
        (b.stops === 1 ? '' : 's') + ', last ' + self.esc(when) +
        (asked.length ? ' — ' + self.esc(asked.join(', ')) : '') + '</div></div>';
      html += '<button class="config-btn" data-unlock-model="' +
        self.esc(b.model) + '" data-unlock-node="' + self.esc(b.nodeId) +
        '">Unlock</button>';
      html += '</div>';
    });
    return html + '</div>';
  },

  async unlockModel(model, nodeId) {
    if (!model) return;
    if (!confirm('Unlock ' + model + '?\n\nThe guard stopped it because the ' +
                 'node ran out of memory. Dropping that record does not change ' +
                 'what the launch will ask for — if nothing else has changed, ' +
                 'it will run out again.')) return;
    try {
      var resp = await fetch('/api/cluster/measurements/forget-stops', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: model, node_id: nodeId || undefined }),
      });
      var data = await resp.json().catch(function () { return {}; });
      if (!resp.ok || data.error) {
        this.toast(data.error || 'Could not unlock', 'error');
        return;
      }
      this.toast('Unlocked ' + model, 'success');
      this.renderConfigMemory();
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
    }
  },

  async _saveMemoryGuard(body) {
    try {
      var resp = await fetch('/api/cluster/safety/memory', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      var data = await resp.json().catch(function () { return {}; });
      var failed = ((data && data.results) || []).filter(function (r) {
        return !r.ok;
      });
      if (!resp.ok || data.error) {
        this.toast(data.error || 'Could not save the reserve', 'error');
      } else if (failed.length) {
        this.toast('Not saved on: ' + failed.map(function (r) {
          return r.node_id;
        }).join(', '), 'error');
      } else {
        this.toast('Memory reserve saved', 'success');
      }
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
    }
    this.renderConfigMemory();
  },

  // ----- Updates ------------------------------------------------------------
  // This deployment updates from source, not from a published image: git pull
  // on the head, then scripts/update-cluster.sh, which builds, distributes
  // and restarts. So the question is not "is there a newer tag" but "is our
  // fork's branch ahead of the commit this image was built from".

  async renderConfigUpdates() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading…</div>';
    var self = this;
    var data = await this.fetchJSON('/api/update/settings');
    if (!data) {
      mount.innerHTML = '<div class="config-empty">Unable to load.</div>';
      return;
    }
    var s = data.settings || {};
    var state = this.state.updateState || {};

    var html = '';
    html += '<h2 class="config-section-title">Updates</h2>';
    html += '<p class="config-section-desc">This cluster builds on the head ' +
      'rather than pulling a published image, so an update is a ' +
      '<code>git pull</code> followed by <code>scripts/update-cluster.sh</code>: ' +
      'build here, distribute to the peers, restart members first and the ' +
      'head last. Checked hourly against the branch below.</p>';

    html += '<div class="config-card">';
    html += '<h3 class="config-card-title">Status</h3>';
    html += '<p class="config-card-desc">' + this.updateStatusLine(state, data) +
      '</p>';
    if (state.commits && state.commits.length) {
      html += '<div class="plan-notes">' + state.commits.map(function (c) {
        return '<div><code>' + self.esc(c.sha) + '</code> ' +
          self.esc(c.subject) + '</div>';
      }).join('') + '</div>';
    }
    if (data.why_not) {
      html += '<div class="plan-warn" style="white-space:pre-wrap">' +
        this.esc(data.why_not) + '</div>';
    }
    // A successful update ends by restarting this container, so the page that
    // watched it is gone by the time it finishes. The outcome is read back
    // from disk and said here, or the update looks like it never ran.
    if (state.last_run && !state.last_run.running) {
      var lr = state.last_run;
      var when = lr.finished_at
        ? new Date(lr.finished_at * 1000).toLocaleString() : '';
      var cls = lr.status === 'done' ? 'plan-notes' : 'plan-warn';
      var text = lr.status === 'done'
        ? 'Last update finished' + (when ? ' ' + when : '') +
          (lr.nodes && lr.nodes.length ? ' — ' + lr.nodes.join(', ') : '') + '.'
        : 'Last update ' + lr.status + (when ? ' ' + when : '') +
          (lr.error ? ': ' + lr.error : '') + '.';
      html += '<div class="' + cls + '">' + this.esc(text) + '</div>';
    }
    html += '<div class="config-actions">' +
      '<button class="config-btn" id="upd-check">Check for updates</button>' +
      '<button class="config-btn" id="upd-run"' +
      (data.can_run ? '' : ' disabled') + '>Update the cluster</button>' +
      '<span class="config-field-hint" id="upd-hint"></span>' +
      '</div></div>';

    html += '<div class="config-card">';
    html += '<h3 class="config-card-title">Where updates come from</h3>';
    html += '<div class="config-form-grid">';
    html += this._field('Repository', 'source_repo', s.source_repo,
                        { hint: 'owner/name on GitHub — your fork, not upstream' });
    html += this._field('Branch', 'source_branch', s.source_branch, {});
    html += this._field('Checkout on the head', 'source_dir', s.source_dir,
                        { hint: 'where git pull runs; mounted into the container as ' +
                                (data.container_source_dir || '/ainode-src') });
    html += '</div>';
    html += '<label class="config-field-label" style="margin-top:12px">Peer nodes (SSH)</label>';
    html += '<input class="form-input" id="upd-nodes" value="' +
      this.esc((s.cluster_ssh_nodes || []).join(', ')) + '" ' +
      'placeholder="Spark2, Spark3">';
    html += '<div class="config-field-hint">Exactly as <code>ssh &lt;name&gt;</code> ' +
      'would take them. Held here rather than taken from discovery: an SSH ' +
      'name and a fabric address are different things, and only you know the ' +
      'mapping. The head is not listed — it is this node.</div>';
    html += '<div class="config-actions">' +
      '<button class="config-btn" id="upd-save">Save</button></div>';
    html += '</div>';

    html += '<div class="config-card" id="upd-log-card" style="display:none">' +
      '<h3 class="config-card-title">Output</h3>' +
      '<pre class="update-log" id="upd-log"></pre></div>';

    mount.innerHTML = html;

    document.getElementById('upd-check').addEventListener('click', function () {
      self.checkForSourceUpdate(true).then(function () {
        self.renderConfigUpdates();
      });
    });
    document.getElementById('upd-save').addEventListener('click', function () {
      self.saveUpdateSettings();
    });
    var run = document.getElementById('upd-run');
    if (run) run.addEventListener('click', function () { self.runSourceUpdate(); });
    // Ask the server whether one is running, rather than trusting a flag
    // this tab happens to hold: a reload during a twenty-minute update used
    // to leave the panel looking idle while the build went on without it.
    this.resumeUpdateJob();
  },

  // Show the output of a running — or just-finished — update, whether or not
  // this tab is the one that started it.
  async resumeUpdateJob() {
    var job = await this.fetchJSON('/api/update/status');
    if (!job || !job.status || job.status === 'idle') return;
    var card = document.getElementById('upd-log-card');
    if (card) card.style.display = '';
    var log = document.getElementById('upd-log');
    if (log) {
      log.textContent = (job.lines || []).join('\n');
      log.scrollTop = log.scrollHeight;
    }
    if (job.running && !this._updateJobPolling) {
      this._updateJobPolling = true;
      this.pollUpdateJob();
    }
  },

  updateStatusLine(state, settings) {
    var repo = (settings && settings.settings && settings.settings.source_repo)
      || (state && state.repo) || '';
    if (state && state.error) {
      return 'Could not check ' + this.esc(repo) + ': ' + this.esc(state.error);
    }
    if (!state || !state.latest) {
      return 'Not checked yet.';
    }
    if (!state.update_available) {
      return 'Up to date with ' + this.esc(repo) + '@' +
        this.esc(state.branch || 'main') + ' (' +
        this.esc((state.current || '').slice(0, 8)) + ').';
    }
    return '<strong>' + state.behind + ' commit' +
      (state.behind === 1 ? '' : 's') + ' behind</strong> ' + this.esc(repo) +
      '@' + this.esc(state.branch || 'main') + ' — running ' +
      this.esc((state.current || '').slice(0, 8)) + ', latest ' +
      this.esc((state.latest || '').slice(0, 8)) + '.';
  },

  // Hourly, and on demand. A manual check passes force, because a "check now"
  // that returns a cached answer is not a check.
  async checkForSourceUpdate(force) {
    var data = await this.fetchJSON('/api/update/check' + (force ? '?force=1' : ''));
    if (data) {
      this.state.updateState = data;
      this.renderUpdateBanner();
    }
    return data;
  },

  renderUpdateBanner() {
    var state = this.state.updateState;
    var mount = document.getElementById('update-banner');
    if (!mount) return;
    // A running update outranks the offer of one. It takes twenty minutes and
    // ends by restarting every node; "3 commits behind" is not the thing to
    // be saying while that happens — and a browser reloaded in the middle of
    // one had nothing at all to tell it the build was still going.
    if (this.state.updateRunning) {
      mount.style.display = '';
      mount.innerHTML =
        '<span>⟳ <strong>Update running</strong> — every node restarts when ' +
        'it finishes, this one last</span>' +
        '<button class="btn-ghost server-btn-sm" id="update-banner-go">' +
        'Output</button>';
    } else if (!state || !state.update_available) {
      mount.innerHTML = '';
      mount.style.display = 'none';
      return;
    } else {
      mount.style.display = '';
      mount.innerHTML =
        '<span>⬆ <strong>' + state.behind + ' commit' +
        (state.behind === 1 ? '' : 's') + '</strong> behind ' +
        this.esc(state.repo || '') + '@' + this.esc(state.branch || 'main') +
        '</span><button class="btn-ghost server-btn-sm" id="update-banner-go">' +
        'Updates</button>';
    }
    var self = this;
    var go = document.getElementById('update-banner-go');
    if (go) {
      go.addEventListener('click', function () {
        self.state.configSection = 'updates';
        self.navigate('config');
        self._configViewInitialized = true;
        self.renderConfig();
      });
    }
  },

  async saveUpdateSettings() {
    var nodes = (document.getElementById('upd-nodes') || {}).value || '';
    var body = {
      source_repo: (document.getElementById('cfg-f-source_repo') || {}).value || '',
      source_branch: (document.getElementById('cfg-f-source_branch') || {}).value || '',
      source_dir: (document.getElementById('cfg-f-source_dir') || {}).value || '',
      cluster_ssh_nodes: nodes.split(/[,\s]+/).filter(function (n) { return n; }),
    };
    try {
      var resp = await fetch('/api/update/settings', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      var data = await resp.json().catch(function () { return {}; });
      if (!resp.ok || data.error) {
        this.toast(data.error || 'Could not save', 'error');
        return;
      }
      this.toast('Saved', 'success');
      // The repo may have changed, so the last answer is about the old one.
      this.state.updateState = null;
      await this.checkForSourceUpdate(true);
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
    }
    this.renderConfigUpdates();
  },

  async runSourceUpdate() {
    var nodes = ((document.getElementById('upd-nodes') || {}).value || '')
      .split(/[,\s]+/).filter(function (n) { return n; });
    if (!confirm('Update the cluster?\n\ngit pull, then build here and ' +
                 'distribute to ' + (nodes.join(', ') || 'no peers') +
                 '.\n\nEvery node restarts, so loaded models are unloaded — ' +
                 'save a profile first if you want them back. This node ' +
                 'restarts LAST and its web UI will drop for a moment when ' +
                 'it does.')) return;
    try {
      var resp = await fetch('/api/update/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes: nodes }),
      });
      var data = await resp.json().catch(function () { return {}; });
      if (!resp.ok || data.error) {
        this.toast(data.error || 'Could not start', 'error');
        return;
      }
      this._updateJobPolling = true;
      this.pollUpdateJob();
    } catch (err) {
      this.toast('Error: ' + err.message, 'error');
    }
  },

  async pollUpdateJob() {
    var card = document.getElementById('upd-log-card');
    var log = document.getElementById('upd-log');
    var hint = document.getElementById('upd-hint');
    if (card) card.style.display = '';
    var job = await this.fetchJSON('/api/update/status');
    if (job && log) {
      log.textContent = (job.lines || []).join('\n');
      log.scrollTop = log.scrollHeight;
    }
    if (hint && job) {
      hint.textContent = job.running ? 'running…'
        : (job.status === 'done' ? 'done' : (job.error || job.status || ''));
    }
    // The head restarts last, which from in here means this container stops
    // itself — so the poll ending in a failed fetch is the expected way for a
    // successful update to finish.
    if (job && job.running) {
      var self = this;
      setTimeout(function () { self.pollUpdateJob(); }, 2000);
    } else {
      this._updateJobPolling = false;
      if (job && job.status === 'done') this.checkForSourceUpdate(true);
    }
  },

  // ----- Training Defaults --------------------------------------------------
  async renderConfigTrainingDefaults() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading…</div>';
    var cfg = await this.fetchJSON('/api/config');
    this.state.configData.config = cfg;
    if (!cfg) { mount.innerHTML = '<div class="config-empty">Unable to load config.</div>'; return; }
    var self = this;
    var method = cfg.training_default_method || 'lora';
    var html = '';
    html += '<h2 class="config-section-title">Training Defaults</h2>';
    html += '<p class="config-section-desc">Default values prefilled when starting a new fine-tuning run.</p>';
    html += '<div class="config-card"><div class="config-form-grid">';
    html += '<div><label class="config-field-label">Default method</label>';
    html += '<select class="form-select" id="cfg-f-training_default_method">';
    ['lora', 'qlora', 'full'].forEach(function (m) {
      html += '<option value="' + m + '"' + (method === m ? ' selected' : '') + '>' + m.toUpperCase() + '</option>';
    });
    html += '</select></div>';
    html += this._field('Default epochs', 'training_default_epochs', cfg.training_default_epochs, { type: 'number' });
    html += this._field('Default batch size', 'training_default_batch_size', cfg.training_default_batch_size, { type: 'number' });
    html += this._field('Default learning rate', 'training_default_learning_rate', cfg.training_default_learning_rate, { type: 'number', step: '0.00001' });
    html += '</div>';
    html += '<div class="config-actions"><button class="config-btn" id="cfg-training-save">Save</button></div>';
    html += '</div>';
    mount.innerHTML = html;
    document.getElementById('cfg-training-save').addEventListener('click', function () {
      self._patchConfig({
        training_default_method: document.getElementById('cfg-f-training_default_method').value,
        training_default_epochs: parseInt(document.getElementById('cfg-f-training_default_epochs').value, 10),
        training_default_batch_size: parseInt(document.getElementById('cfg-f-training_default_batch_size').value, 10),
        training_default_learning_rate: parseFloat(document.getElementById('cfg-f-training_default_learning_rate').value),
      });
    });
  },

  // ----- Network ------------------------------------------------------------
  async renderConfigNetwork() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading…</div>';
    var cfg = await this.fetchJSON('/api/config');
    this.state.configData.config = cfg;
    if (!cfg) { mount.innerHTML = '<div class="config-empty">Unable to load config.</div>'; return; }
    var self = this;
    var host = cfg.host || '0.0.0.0';
    var html = '';
    html += '<h2 class="config-section-title">Network</h2>';
    html += '<p class="config-section-desc">Port and binding configuration. <strong>Changes to ports require an AINode restart to take effect.</strong></p>';
    html += '<div class="config-card"><div class="config-form-grid">';
    html += this._field('API port (vLLM)', 'api_port', cfg.api_port, { type: 'number' });
    html += this._field('Web port (UI + proxy)', 'web_port', cfg.web_port, { type: 'number' });
    html += this._field('Discovery port (UDP)', 'discovery_port', cfg.discovery_port, { type: 'number' });
    html += '<div><label class="config-field-label">Bind host</label>';
    html += '<select class="form-select" id="cfg-f-host">';
    [['0.0.0.0', 'All interfaces (0.0.0.0)'], ['127.0.0.1', 'Localhost only (127.0.0.1)']].forEach(function (pair) {
      html += '<option value="' + pair[0] + '"' + (host === pair[0] ? ' selected' : '') + '>' + pair[1] + '</option>';
    });
    html += '</select></div>';
    html += this._field('CORS origins', 'cors_origins', cfg.cors_origins || '', { hint: 'Comma-separated list of allowed origins' });
    html += '</div>';
    html += '<div class="config-actions"><button class="config-btn" id="cfg-net-save">Save</button></div>';
    html += '</div>';
    mount.innerHTML = html;
    document.getElementById('cfg-net-save').addEventListener('click', function () {
      self._patchConfig({
        api_port: parseInt(document.getElementById('cfg-f-api_port').value, 10),
        web_port: parseInt(document.getElementById('cfg-f-web_port').value, 10),
        discovery_port: parseInt(document.getElementById('cfg-f-discovery_port').value, 10),
        host: document.getElementById('cfg-f-host').value,
        cors_origins: document.getElementById('cfg-f-cors_origins').value,
      }, { restartHint: true });
    });
  },

  // ----- Monitoring (MQTT telemetry) ---------------------------------------

  async renderConfigMonitoring() {
    var mount = this._configMount();
    if (!mount) return;
    mount.innerHTML = '<div class="config-empty">Loading…</div>';
    var data = await this.fetchJSON('/api/telemetry/mqtt');
    if (!data) {
      mount.innerHTML = '<div class="config-empty">Unable to load telemetry settings.</div>';
      return;
    }
    var self = this;
    var s = data.settings || {};
    var status = data.status || {};

    var html = '';
    html += '<h2 class="config-section-title">Monitoring</h2>';
    html += '<p class="config-section-desc">Publish this node\'s metrics to an MQTT broker — CPU, memory, disk, per-interface network load, GPU, and what each loaded model is doing. ' +
            'The head also publishes the cluster view. Anything that speaks MQTT can read it: Home Assistant, Node-RED, Telegraf into Grafana.</p>';

    // Live status first: whether it is actually publishing is the question
    // someone opening this page has.
    html += '<div class="config-card">';
    html += '  <h3 class="config-card-title">Status</h3>';
    html += '  <div class="config-form-grid">';
    html += '    <div><div class="config-field-label">Publishing</div><div>' +
            (status.running ? '<span style="color:var(--nvidia-green)">yes</span>' : 'no') + '</div></div>';
    html += '    <div><div class="config-field-label">Broker</div><div class="mono">' + this.esc(status.broker || '—') + '</div></div>';
    html += '    <div><div class="config-field-label">Messages sent</div><div>' + (status.published || 0) + '</div></div>';
    if (status.last_error) {
      html += '    <div><div class="config-field-label">Last error</div><div style="color:#ff6b6b">' + this.esc(status.last_error) + '</div></div>';
    }
    html += '  </div>';
    html += '  <div class="config-actions"><button class="config-btn" id="cfg-mqtt-refresh">Refresh status</button></div>';
    html += '</div>';

    html += '<div class="config-card"><div class="config-form-grid">';
    html += '<div><label class="config-field-label">Publish telemetry</label>' +
            '<label style="display:flex;align-items:center;gap:8px;margin-top:6px">' +
            '<input type="checkbox" id="cfg-mqtt-enabled"' + (s.mqtt_enabled ? ' checked' : '') + '> enabled</label></div>';
    html += this._field('Broker host', 'mqtt_host', s.mqtt_host, { hint: 'IP or hostname of the MQTT server' });
    html += this._field('Broker port', 'mqtt_port', s.mqtt_port, { type: 'number' });
    html += this._field('Username', 'mqtt_username', s.mqtt_username, { hint: 'Leave empty for an anonymous broker' });
    html += '<div><label class="config-field-label" for="cfg-f-mqtt_password">Password</label>' +
            '<input class="form-input" id="cfg-f-mqtt_password" type="password" placeholder="' +
            (data.password_set ? 'stored — leave empty to keep it' : 'none stored') + '">' +
            '<div class="config-field-hint">Kept in the secrets store, not in config.json.</div></div>';
    html += this._field('Topic prefix', 'mqtt_topic_prefix', s.mqtt_topic_prefix,
                        { hint: 'Topics become prefix/node-id/system, /gpu, /models — plus prefix/cluster from the head' });
    html += this._field('Interval (seconds)', 'mqtt_interval', s.mqtt_interval,
                        { type: 'number', hint: 'How often to publish. 1-3600 — one second is for watching something happen, not for leaving on.' });
    html += '<div><label class="config-field-label">TLS</label>' +
            '<label style="display:flex;align-items:center;gap:8px;margin-top:6px">' +
            '<input type="checkbox" id="cfg-mqtt-tls"' + (s.mqtt_tls ? ' checked' : '') + '> use TLS</label></div>';
    html += '<div><label class="config-field-label">Retain</label>' +
            '<label style="display:flex;align-items:center;gap:8px;margin-top:6px">' +
            '<input type="checkbox" id="cfg-mqtt-retain"' + (s.mqtt_retain ? ' checked' : '') + '> keep the last message on the broker</label>' +
            '<div class="config-field-hint">Handy after a broker restart; a retained message from a node that has gone away still looks alive.</div></div>';
    html += '<div><label class="config-field-label">Logs</label>' +
            '<label style="display:flex;align-items:center;gap:8px;margin-top:6px">' +
            '<input type="checkbox" id="cfg-mqtt-logs"' + (s.mqtt_logs ? ' checked' : '') + '> ' +
            'forward log lines</label>' +
            '<div class="config-field-hint">This node\'s own log to ' +
            '<code>logs/ainode</code>, and each engine instance to ' +
            '<code>logs/vllm/&lt;model&gt;</code>. Only what is new since the ' +
            'last publish, progress bars dropped. Off by default — a vLLM log ' +
            'is a firehose.</div></div>';
    html += this._field('Log lines per message', 'mqtt_log_lines', s.mqtt_log_lines,
                        { type: 'number', hint: 'Older lines are dropped and the payload says how many. 1-1000.' });
    html += '<div><label class="config-field-label">Log level (this node)</label>' +
            '<select class="form-select" id="cfg-f-mqtt_log_level">' +
            ['DEBUG', 'INFO', 'WARNING', 'ERROR'].map(function (lv) {
              return '<option value="' + lv + '"' +
                ((s.mqtt_log_level || 'INFO') === lv ? ' selected' : '') + '>' +
                lv + '</option>';
            }).join('') + '</select>' +
            '<div class="config-field-hint">Applies to AINode\'s own lines. The engine does not level its output, so its log is forwarded as written.</div></div>';
    html += '</div>';
    html += '<div class="config-actions">' +
            '<button class="config-btn" id="cfg-mqtt-save">Save</button>' +
            '<button class="config-btn" id="cfg-mqtt-test">Test connection</button>' +
            '<button class="config-btn" id="cfg-mqtt-publish">Publish now</button>' +
            '<button class="config-btn" id="cfg-mqtt-preview">Show payload</button>' +
            '<button class="config-btn" id="cfg-mqtt-cluster">Apply to all nodes</button>' +
            '</div>';
    html += '<div class="config-card-desc" style="margin-top:8px">Each node ' +
            'publishes its own CPU, memory, disk and network — no other node ' +
            'can see those. Configured here alone, you get telemetry from this ' +
            'node only. <strong>Apply to all nodes</strong> copies these ' +
            'settings, password included, to every node over the cluster ' +
            'network.</div>';
    html += '<div id="cfg-mqtt-result" class="config-card-desc" style="margin-top:10px"></div>';
    html += '</div>';

    html += '<div class="config-card">';
    html += '  <h3 class="config-card-title">Topics</h3>';
    html += '  <ul style="margin:0;padding-left:18px;font-family:var(--font-mono);font-size:12px">';
    (data.topics || []).forEach(function (t) {
      html += '<li>' + self.esc(t) + '</li>';
    });
    html += '  </ul>';
    html += '  <p class="config-card-desc" style="margin-top:10px">One JSON message per topic. <code>cluster</code> is published by the head only — a member would overwrite the complete picture with its own partial one.</p>';
    html += '</div>';

    mount.innerHTML = html;

    var out = document.getElementById('cfg-mqtt-result');
    var say = function (text, bad) {
      out.innerHTML = '<span style="color:' + (bad ? '#ff6b6b' : 'var(--nvidia-green)') + '">' + self.esc(text) + '</span>';
    };

    // Explicit, because the page no longer rebuilds itself under the cursor.
    document.getElementById('cfg-mqtt-refresh').addEventListener('click', function () {
      self.renderConfigMonitoring();
    });

    document.getElementById('cfg-mqtt-save').addEventListener('click', async function () {
      var body = {
        mqtt_enabled: document.getElementById('cfg-mqtt-enabled').checked,
        mqtt_tls: document.getElementById('cfg-mqtt-tls').checked,
        mqtt_retain: document.getElementById('cfg-mqtt-retain').checked,
        mqtt_host: document.getElementById('cfg-f-mqtt_host').value.trim(),
        mqtt_port: parseInt(document.getElementById('cfg-f-mqtt_port').value, 10),
        mqtt_username: document.getElementById('cfg-f-mqtt_username').value.trim(),
        mqtt_topic_prefix: document.getElementById('cfg-f-mqtt_topic_prefix').value.trim(),
        mqtt_interval: parseInt(document.getElementById('cfg-f-mqtt_interval').value, 10),
        mqtt_password: document.getElementById('cfg-f-mqtt_password').value,
        mqtt_logs: document.getElementById('cfg-mqtt-logs').checked,
        mqtt_log_lines: parseInt(document.getElementById('cfg-f-mqtt_log_lines').value, 10),
        mqtt_log_level: document.getElementById('cfg-f-mqtt_log_level').value,
      };
      var resp = await fetch('/api/telemetry/mqtt', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      var d = await resp.json().catch(function () { return {}; });
      if (!resp.ok) { say(d.error || 'Could not save', true); return; }
      self.toast('Monitoring settings saved', 'success');
      self.renderConfigMonitoring();
    });

    document.getElementById('cfg-mqtt-cluster').addEventListener('click', async function () {
      if (!confirm('Copy these settings to every other node?\n\nThe broker ' +
                   'password travels over the cluster network in the clear.')) return;
      say('Applying…');
      var resp = await fetch('/api/telemetry/mqtt/apply-to-cluster', { method: 'POST' });
      var d = await resp.json().catch(function () { return {}; });
      if (d.error) { say(d.error, true); return; }
      var lines = (d.results || []).map(function (r) {
        return (r.ok ? '✓ ' : '✕ ') + r.node + (r.error ? ': ' + r.error : '');
      });
      out.innerHTML = lines.map(function (l) {
        return '<div class="profile-report-line' + (l.charAt(0) === '✕' ? ' error' : '') +
               '">' + self.esc(l) + '</div>';
      }).join('') || '<span>No other nodes.</span>';
    });

    document.getElementById('cfg-mqtt-test').addEventListener('click', async function () {
      say('Connecting…');
      var resp = await fetch('/api/telemetry/mqtt/test', { method: 'POST' });
      var d = await resp.json().catch(function () { return {}; });
      if (d.ok) say('Connected to ' + d.broker);
      else say(d.error || 'Connection failed', true);
    });

    document.getElementById('cfg-mqtt-publish').addEventListener('click', async function () {
      say('Publishing…');
      var resp = await fetch('/api/telemetry/mqtt/publish', { method: 'POST' });
      var d = await resp.json().catch(function () { return {}; });
      if (d.ok) say('Published ' + d.published + ' message(s)');
      else say(d.error || 'Publish failed', true);
    });

    document.getElementById('cfg-mqtt-preview').addEventListener('click', async function () {
      say('Sampling…');
      var d = await self.fetchJSON('/api/telemetry/preview');
      if (!d) { say('Could not build a preview', true); return; }
      out.innerHTML = '<pre style="max-height:420px;overflow:auto;font-size:11px;background:rgba(0,0,0,.35);padding:10px;border-radius:6px">' +
        self.esc(JSON.stringify(d.payloads, null, 2)) + '</pre>';
    });
  },

  // ----- About --------------------------------------------------------------
  async renderConfigAbout() {
    var mount = this._configMount();
    if (!mount) return;
    var status = await this.fetchJSON('/api/status');
    var gpu = status && status.gpu;
    var html = '';
    html += '<h2 class="config-section-title">About</h2>';
    html += '<p class="config-section-desc">AINode — local AI platform powered by argentos.ai.</p>';
    html += '<div class="config-card"><div class="config-form-grid">';
    html += '<div><div class="config-field-label">AINode version</div><div>' + this.esc((status && status.version) || 'n/a') + '</div></div>';
    html += '<div><div class="config-field-label">Node ID</div><div class="mono" style="font-family:var(--font-mono);font-size:12px">' + this.esc((status && status.node_id) || 'n/a') + '</div></div>';
    html += '<div><div class="config-field-label">Current model</div><div>' + this.esc((status && status.model) || 'none') + '</div></div>';
    html += '<div><div class="config-field-label">GPU</div><div>' + this.esc(gpu ? (gpu.name + (gpu.memory_total_mb ? ' · ' + Math.round(gpu.memory_total_mb / 1024) + ' GB' : '')) : 'CPU only') + '</div></div>';
    html += '<div><div class="config-field-label">Cluster role</div><div>' + this.esc((status && status.cluster_role) || 'n/a') + '</div></div>';
    html += '<div><div class="config-field-label">Master node</div><div class="mono">' + this.esc((status && status.master_node_id) || '—') + '</div></div>';
    html += '</div></div>';
    html += '<div class="config-card">';
    html += '  <h3 class="config-card-title">Links</h3>';
    html += '  <p><a href="https://ainode.dev" target="_blank" rel="noopener" style="color:var(--nvidia-green)">ainode.dev</a> · ';
    html += '  <a href="https://github.com/getainode/ainode" target="_blank" rel="noopener" style="color:var(--nvidia-green)">GitHub</a> · ';
    html += '  <a href="https://docs.argentos.ai" target="_blank" rel="noopener" style="color:var(--nvidia-green)">Docs</a></p>';
    html += '  <p class="config-card-desc" style="margin-top:10px">Licensed under Apache 2.0. Powered by argentos.ai.</p>';
    html += '</div>';
    mount.innerHTML = html;
  },

  // ----- Helpers ------------------------------------------------------------
  _field(label, key, value, opts) {
    opts = opts || {};
    var type = opts.type || 'text';
    var readonly = opts.readonly ? ' readonly' : '';
    var step = opts.step ? ' step="' + opts.step + '"' : '';
    var mono = opts.mono ? ' style="font-family:var(--font-mono);font-size:12px"' : '';
    var val = value === null || value === undefined ? '' : String(value);
    var html = '<div>';
    html += '  <label class="config-field-label" for="cfg-f-' + key + '">' + this.esc(label) + '</label>';
    html += '  <input class="form-input" id="cfg-f-' + key + '" type="' + type + '"' + step + readonly + mono + ' value="' + this.esc(val) + '">';
    if (opts.hint) html += '<div class="config-field-hint">' + this.esc(opts.hint) + '</div>';
    html += '</div>';
    return html;
  },

  async _patchConfig(patch, opts) {
    opts = opts || {};
    // Strip NaN / empty numeric
    Object.keys(patch).forEach(function (k) {
      var v = patch[k];
      if (typeof v === 'number' && Number.isNaN(v)) delete patch[k];
      if (v === '') patch[k] = null;
    });
    var resp = await fetch('/api/config', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    var body = await resp.json().catch(function () { return {}; });
    if (!resp.ok) {
      this.toast((body.error && body.error.message) || 'Save failed', 'error');
      return;
    }
    if (opts.restartHint) {
      this.toast('Saved — restart AINode for network changes to apply', 'info');
    } else {
      this.toast('Saved', 'success');
    }
  },
});
