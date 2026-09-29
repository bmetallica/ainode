/* AINode Command Center — the Downloads view: search, downloads, resume, import.
 *
 * Methods of the AINode object, split out of app.js by view (W2 in
 * upgrade-fixes.md) so that work on one view stops conflicting with work
 * on every other. A classic script loaded after app.js: the methods are
 * the same, called the same way, with the same `this`.
 */
Object.assign(AINode, {
  // ========================================================================
  //  DOWNLOADS VIEW (center-stage)
  // ========================================================================

  renderLiveCatalog(container, loaded, gpuMem, filterPills, source) {
    var self = this;
    var titles = {
      trending: '🔥 Trending Models',
      openrouter: '🚀 Most Used in Production',
      latest: '✨ Latest Releases',
    };
    var subtitles = {
      trending: 'Hot on HuggingFace right now',
      openrouter: 'Ranked by real API traffic on OpenRouter',
      latest: 'Newest text-generation models on HuggingFace',
    };

    // Clear if switching from another mode
    if (container.querySelector('#downloads-search') || container.querySelector('#hf-search-input')) {
      container.innerHTML = '';
    }

    var needsToolbar = !container.querySelector('#live-catalog-title');
    if (needsToolbar) {
      container.innerHTML =
        '<div class="downloads-header">' +
        '<h2 class="view-title" id="live-catalog-title">' + titles[source] + '</h2>' +
        '<div class="downloads-count" id="live-count">Loading...</div>' +
        '</div>' +
        '<div id="downloads-queue" class="downloads-queue"></div>' +
        '<div class="downloads-toolbar">' +
        '<div class="live-catalog-subtitle">' + subtitles[source] + '</div>' +
        '<div class="pill-group downloads-filters" id="downloads-filters">' + filterPills + '</div>' +
        '</div>' +
        '<div id="downloads-results"><div class="downloads-empty">Fetching live data...</div></div>';
    } else {
      var titleEl = container.querySelector('#live-catalog-title');
      if (titleEl) titleEl.textContent = titles[source];
      var subEl = container.querySelector('.live-catalog-subtitle');
      if (subEl) subEl.textContent = subtitles[source];
      var pillsEl = container.querySelector('#downloads-filters');
      if (pillsEl) pillsEl.innerHTML = filterPills;
    }

    var resultsContainer = container.querySelector('#downloads-results');
    var countEl = container.querySelector('#live-count');

    // Rebind filter pills
    container.querySelectorAll('.downloads-filter').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.state.modelsFilter = btn.dataset.filter;
        self.renderDownloads();
      });
    });

    fetch('/api/models/' + source)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var models = data.models || [];
        if (countEl) countEl.textContent = models.length + ' models';
        if (!resultsContainer) return;
        if (models.length === 0) {
          resultsContainer.innerHTML = '<div class="downloads-empty">No data available from ' + source + '.</div>';
          return;
        }
        var rows = models.map(function (m) {
          var isLoaded = loaded.includes(m.hf_repo);
          var isOnDisk = isLoaded || !!(self.state.downloadedModels && self.state.downloadedModels[m.hf_repo]);
          var sizeStr = m.size_gb > 0 ? '~' + Math.round(m.size_gb) + ' GB' : 'size unknown';
          var paramsStr = m.params_b ? m.params_b + 'B params' : '';
          var fits = gpuMem > 0 && m.size_gb > 0 && gpuMem >= (m.min_memory_gb || m.size_gb);
          var fitBadge = (gpuMem > 0 && m.size_gb > 0) ? (fits ?
            '<span class="fit-badge fits">Fits GPU</span>' :
            '<span class="fit-badge no-fit">Too large</span>') : '';
          var recBadge = m.recommended ? '<span class="fit-badge rec">Recommended</span>' : '';
          var quantBadge = m.quantization ? '<span class="fit-badge quant">' + self.esc(m.quantization.toUpperCase()) + '</span>' : '';
          var capBadges = self.renderCapabilityBadges(m);
          var statusBadge = isLoaded ?
            '<span class="model-badge loaded">Loaded</span>' :
            isOnDisk ?
            '<span class="model-badge loaded">Downloaded</span>' :
            '<span class="model-badge available">Available</span>';
          var ageStr = m.created_at ? self.relativeTime(m.created_at) : '';
          var downloadsStr = m.downloads ? self.formatNumber(m.downloads) + ' ⬇' : '';
          var likesStr = m.likes ? '❤ ' + self.formatNumber(m.likes) : '';
          var metaParts = [paramsStr, sizeStr, ageStr, downloadsStr, likesStr].filter(Boolean);
          var detailsBtn = '<button class="btn-sm download-details-btn" data-info-repo="' + self.esc(m.hf_repo) + '">Details</button>';
          var downloadBtn = isOnDisk ? '' :
            '<button class="btn-sm downloads-download-btn" data-model-id="' + self.esc(m.hf_repo) + '">Download</button>';
          return '<div class="download-card" data-model-id="' + self.esc(m.hf_repo) + '">' +
            '<div class="download-card-main">' +
            '<div class="download-card-info">' +
            '<div class="download-card-header">' +
            '<div class="download-card-name">' + self.esc(m.name || m.hf_repo) + '</div>' +
            '<div class="download-card-badges">' + recBadge + quantBadge + capBadges + fitBadge + statusBadge + '</div>' +
            '</div>' +
            '<div class="download-card-repo">' + self.esc(m.hf_repo) + '</div>' +
            '<div class="download-card-desc">' + metaParts.join(' · ') + '</div>' +
            '</div>' +
            '<div class="download-card-actions">' + detailsBtn + downloadBtn + '</div>' +
            '</div>' +
            '</div>';
        }).join('');
        resultsContainer.innerHTML = '<div class="downloads-grid">' + rows + '</div>';
        self.bindRepoDownloadButtons(resultsContainer);
      })
      .catch(function (err) {
        if (resultsContainer) resultsContainer.innerHTML = '<div class="downloads-empty">Failed to fetch: ' + self.esc(err.message || 'network error') + '</div>';
      });
  },

  bindRepoDownloadButtons(container) {
    var self = this;
    container.querySelectorAll('.downloads-download-btn').forEach(function (btn) {
      if (btn.dataset.bound) return;
      btn.dataset.bound = '1';
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var hfRepo = btn.dataset.modelId;
        if (!hfRepo || hfRepo.indexOf('/') === -1) {
          self.toast('Invalid model repo', 'error');
          return;
        }
        self.startRepoDownload(hfRepo);
      });
    });

    // Details button + capability badges open the detail modal
    container.querySelectorAll('[data-info-repo]').forEach(function (el) {
      if (el.dataset.infoBound) return;
      el.dataset.infoBound = '1';
      el.style.cursor = 'pointer';
      el.addEventListener('click', function (e) {
        e.stopPropagation();
        var repo = el.dataset.infoRepo;
        if (repo) self.showModelDetail(repo);
      });
    });

    // Delete buttons
    container.querySelectorAll('.downloads-delete-btn').forEach(function (btn) {
      if (btn.dataset.bound) return;
      btn.dataset.bound = '1';
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var repo = btn.dataset.modelId;
        var nodes = (btn.getAttribute('data-nodes') || '').split(',')
          .filter(function (n) { return n; });
        self.confirmDeleteModel(repo, nodes);
      });
    });
  },

  // Which nodes hold this model's weights, from the cluster listing.
  nodesHolding(model) {
    var repo = model.hf_repo || model.id;
    var row = (this.state.downloadedList || []).find(function (m) {
      return (m.hf_repo || m.id) === repo;
    });
    return (row && row.nodes) || (model.nodes || []);
  },

  nodeLabel(nodeId) {
    var names = this.state.modelNodeNames || {};
    return (names[nodeId] || nodeId).replace(/-DGX|-GX10/i, '');
  },

  confirmDeleteModel(hfRepo, nodeIds) {
    var self = this;
    var existing = document.getElementById('confirm-delete-modal');
    if (existing) existing.remove();

    var modal = document.createElement('div');
    modal.id = 'confirm-delete-modal';
    modal.className = 'model-detail-modal-overlay';
    modal.innerHTML =
      '<div class="model-detail-modal" style="max-width:480px">' +
        '<div class="md-header">' +
          '<div class="md-header-left">' +
            '<div class="md-icon" style="color:var(--red);border-color:rgba(255,51,51,0.4)">⚠</div>' +
            '<div class="md-title">Delete Model</div>' +
          '</div>' +
          '<button class="md-close">×</button>' +
        '</div>' +
        '<div class="md-description">' +
          'Permanently remove <strong>' + self.esc(hfRepo) + '</strong> from ' +
          ((nodeIds || []).length
            ? self.esc((nodeIds || []).map(function (n) { return self.nodeLabel(n); })
                .join(' + '))
            : 'disk') + '?' +
          '<br><span style="color:var(--text-muted);font-size:13px">This frees the disk space immediately. You can re-download anytime.</span>' +
        '</div>' +
        '<div class="md-footer">' +
          '<button class="btn-sm" id="cd-cancel" style="background:transparent;color:var(--text-secondary);border:1px solid var(--border-hover)">Cancel</button>' +
          '<button class="btn-sm" id="cd-confirm" style="background:var(--red);color:#fff;border:1px solid var(--red);font-weight:700;letter-spacing:0.5px;padding:10px 22px">DELETE</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(modal);

    var close = function () { modal.remove(); };
    modal.querySelector('.md-close').addEventListener('click', close);
    modal.querySelector('#cd-cancel').addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });

    modal.querySelector('#cd-confirm').addEventListener('click', function () {
      var btn = modal.querySelector('#cd-confirm');
      btn.disabled = true;
      btn.textContent = 'Deleting...';
      // Every node that holds it, not only this one. The list is
      // cluster-wide now, so a Delete that could only reach the head's disk
      // would report "not downloaded" about a model plainly shown as present.
      var targets = (nodeIds || []).length ? nodeIds : [null];
      Promise.all(targets.map(function (nodeId) {
        return fetch('/api/cluster/delete-repo', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(nodeId ? { hf_repo: hfRepo, node_id: nodeId }
                                      : { hf_repo: hfRepo }),
        }).then(function (r) { return r.json(); });
      })).then(function (all) {
        var data = all.find(function (d) { return d && d.error; })
          || all[0] || {};
        if (data.error) {
          self.toast('Delete failed: ' + data.error, 'error');
          btn.disabled = false;
          btn.textContent = 'DELETE';
          return;
        }
        self.toast('Deleted ' + hfRepo + ' (freed ' + (data.freed_gb || '?') + ' GB)', 'success');
        close();
        // Both lists, not just the catalog: the disk view is what just
        // changed, and leaving it to the TTL shows the model as still there
        // for up to a minute after its own delete reported success.
        self.invalidate();
        self.refresh();
        self._downloadsViewInitialized = false;
        self.renderDownloads();
      }).catch(function (err) {
        self.toast('Delete failed: ' + err.message, 'error');
        btn.disabled = false;
        btn.textContent = 'DELETE';
      });
    });
  },

  // Persist active downloads to localStorage so page refresh doesn't lose them
  saveActiveDownloads() {
    try {
      var toSave = {};
      Object.keys(this.state.activeDownloads || {}).forEach(function (repo) {
        var dl = this.state.activeDownloads[repo];
        toSave[repo] = {
          jobId: dl.jobId,
          hfRepo: dl.hfRepo,
          startedAt: dl.startedAt,
          status: dl.status,
          totalBytes: dl.totalBytes || 0,
          downloadedBytes: dl.downloadedBytes || 0,
          progress: dl.progress,
        };
      }, this);
      localStorage.setItem('ainode_active_downloads', JSON.stringify(toSave));
    } catch (e) { /* quota or disabled */ }
  },

  loadActiveDownloads() {
    try {
      var raw = localStorage.getItem('ainode_active_downloads');
      if (!raw) return;
      var saved = JSON.parse(raw) || {};
      this.state.activeDownloads = this.state.activeDownloads || {};
      var self = this;
      Object.keys(saved).forEach(function (repo) {
        var dl = saved[repo];
        if (!dl || !dl.jobId) return;
        // Discard entries older than 12 hours — stale
        if (dl.startedAt && Date.now() - dl.startedAt > 12 * 3600 * 1000) return;
        if (dl.status === 'completed' || dl.status === 'failed') return;
        self.state.activeDownloads[repo] = Object.assign({ elapsed: 0 }, dl);
        self.resumeDownloadPolling(repo);
      });
    } catch (e) { /* ignore */ }
  },

  resumeDownloadPolling(hfRepo) {
    var self = this;
    var dl = self.state.activeDownloads[hfRepo];
    if (!dl || !dl.jobId) return;
    if (dl._pollId) return;  // already polling

    dl._pollId = setInterval(function () {
      self.pollDownloadOnce(hfRepo);
    }, 2000);

    dl._tickId = setInterval(function () {
      var d = self.state.activeDownloads[hfRepo];
      if (!d || d.status !== 'downloading') return;
      d.elapsed = Math.floor((Date.now() - d.startedAt) / 1000);
      self.renderQueueItemInPlace(hfRepo);
    }, 1000);
  },

  pollDownloadOnce(hfRepo) {
    var self = this;
    var dl = self.state.activeDownloads && self.state.activeDownloads[hfRepo];
    if (!dl) return;

    fetch('/api/models/download/status?job_id=' + encodeURIComponent(dl.jobId))
      .then(function (r) { return r.json(); })
      .then(function (st) {
        if (st.error || st.status === 'unknown') {
          // Job vanished — treat as probably completed if the model is downloaded
          self.stopDownloadPolling(hfRepo);
          self.checkIfDownloaded(hfRepo);
          return;
        }
        dl.status = st.status;
        dl.elapsed = Math.floor((Date.now() - dl.startedAt) / 1000);
        dl.totalBytes = st.total_bytes || dl.totalBytes || 0;
        // Monotonic — only go up, never down
        var newBytes = st.downloaded_bytes || 0;
        if (newBytes >= (dl.downloadedBytes || 0)) dl.downloadedBytes = newBytes;
        if (st.progress != null && (dl.progress == null || st.progress >= dl.progress)) {
          dl.progress = st.progress;
        }

        self.saveActiveDownloads();
        self.renderQueueItemInPlace(hfRepo);
        self.updateNavDownloadBadge();

        if (st.status === 'completed') {
          self.stopDownloadPolling(hfRepo);
          self.toast('Downloaded: ' + hfRepo, 'success');
          // The model is on disk now: both lists have to say so on the next
          // render, not a minute later.
          self.invalidate();
          // Keep in queue 8s so user sees completion, then remove
          dl.status = 'completed';
          dl.progress = 100;
          dl.downloadedBytes = dl.totalBytes || dl.downloadedBytes;
          self.renderQueueItemInPlace(hfRepo);
          setTimeout(function () {
            delete self.state.activeDownloads[hfRepo];
            self.saveActiveDownloads();
            self.renderDownloadsQueue();
            self.updateNavDownloadBadge();
          }, 6000);
        } else if (st.status === 'failed') {
          self.stopDownloadPolling(hfRepo);
          self.toast('Download failed: ' + (st.error || 'unknown'), 'error');
          dl.error = st.error;
          self.renderQueueItemInPlace(hfRepo);
          setTimeout(function () {
            delete self.state.activeDownloads[hfRepo];
            self.saveActiveDownloads();
            self.renderDownloadsQueue();
            self.updateNavDownloadBadge();
          }, 10000);
        }
      })
      .catch(function () { /* keep polling on transient errors */ });
  },

  stopDownloadPolling(hfRepo) {
    var dl = this.state.activeDownloads && this.state.activeDownloads[hfRepo];
    if (!dl) return;
    if (dl._pollId) { clearInterval(dl._pollId); dl._pollId = null; }
    if (dl._tickId) { clearInterval(dl._tickId); dl._tickId = null; }
  },

  checkIfDownloaded(hfRepo) {
    var self = this;
    var slug = hfRepo.replace(/\//g, '--').toLowerCase();
    fetch('/api/models/' + encodeURIComponent(slug)).then(function (r) {
      return r.ok ? r.json() : null;
    }).then(function (info) {
      var dl = self.state.activeDownloads && self.state.activeDownloads[hfRepo];
      if (!dl) return;
      if (info && info.downloaded) {
        dl.status = 'completed';
        dl.progress = 100;
        self.toast('Downloaded: ' + hfRepo, 'success');
      } else {
        dl.status = 'failed';
        dl.error = 'Job expired';
      }
      self.renderQueueItemInPlace(hfRepo);
      self.saveActiveDownloads();
      setTimeout(function () {
        delete self.state.activeDownloads[hfRepo];
        self.saveActiveDownloads();
        self.renderDownloadsQueue();
        self.updateNavDownloadBadge();
      }, 6000);
    }).catch(function () { /* ignore */ });
  },

  // Update just one queue item's DOM in place — no re-rendering
  renderQueueItemInPlace(hfRepo) {
    var dl = this.state.activeDownloads && this.state.activeDownloads[hfRepo];
    if (!dl) return;
    var queue = document.getElementById('downloads-queue');
    if (!queue) return;
    if (queue.style.display === 'none' || !queue.children.length) {
      this.renderDownloadsQueue();
      return;
    }
    var existing = queue.querySelector('[data-queue-repo="' + CSS.escape(hfRepo) + '"]');
    if (!existing) {
      this.renderDownloadsQueue();
      return;
    }
    existing.outerHTML = this.renderQueueItem(dl);
  },

  // Reconcile on startup — ask server for active jobs that might be ours
  reconcileActiveDownloads() {
    var self = this;
    fetch('/api/models/downloads/active')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var jobs = (data.jobs || []).filter(function (j) {
          return j.status === 'downloading';
        });
        jobs.forEach(function (job) {
          var repo = job.model_id;
          if (!repo || !repo.includes('/')) return;
          if (self.state.activeDownloads && self.state.activeDownloads[repo]) return;
          self.state.activeDownloads = self.state.activeDownloads || {};
          self.state.activeDownloads[repo] = {
            jobId: job.job_id,
            hfRepo: repo,
            status: 'downloading',
            startedAt: Date.now() - 1000,
            elapsed: 0,
            totalBytes: job.total_bytes || 0,
            downloadedBytes: job.downloaded_bytes || 0,
            progress: job.progress,
          };
          self.resumeDownloadPolling(repo);
          self.toast('Resumed tracking: ' + repo, 'info');
        });
        self.saveActiveDownloads();
        self.renderDownloadsQueue();
        self.updateNavDownloadBadge();
      }).catch(function () { /* ignore */ });
  },

  startRepoDownload(hfRepo) {
    var self = this;
    if (!self.state.activeDownloads) self.state.activeDownloads = {};

    if (self.state.activeDownloads[hfRepo] &&
        self.state.activeDownloads[hfRepo].status === 'downloading') {
      self.toast('Already downloading ' + hfRepo, 'info');
      self.navigate('downloads');
      return;
    }

    // Guard: already on disk — offer to launch instead of re-downloading
    if (self.state.downloadedModels && self.state.downloadedModels[hfRepo]) {
      self.toast(hfRepo + ' is already downloaded. Use Launch to load it.', 'info');
      return;
    }

    fetch('/api/models/download-repo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hf_repo: hfRepo }),
    }).then(function (resp) {
      if (!resp.ok) return resp.text().then(function (t) { throw new Error('HTTP ' + resp.status + ': ' + t.slice(0, 80)); });
      return resp.json();
    }).then(function (data) {
      if (data.error) { self.toast(data.error, 'error'); return; }

      self.state.activeDownloads[hfRepo] = {
        jobId: data.job_id,
        status: 'downloading',
        startedAt: Date.now(),
        elapsed: 0,
        hfRepo: hfRepo,
        totalBytes: 0,
        downloadedBytes: 0,
        progress: null,
      };
      self.saveActiveDownloads();

      self.navigate('downloads');
      self.renderDownloadsQueue();
      self.toast('Downloading ' + hfRepo, 'info');
      self.resumeDownloadPolling(hfRepo);
      self.updateNavDownloadBadge();
    }).catch(function (err) {
      self.toast('Download failed: ' + err.message, 'error');
    });
  },

  renderDownloadsQueue() {
    var container = document.getElementById('downloads-queue');
    if (!container) return;
    var self = this;
    var active = Object.values(this.state.activeDownloads || {});
    if (active.length === 0) {
      container.innerHTML = '';
      container.style.display = 'none';
      return;
    }
    container.style.display = '';
    container.innerHTML =
      '<div class="queue-header">' +
        '<span class="queue-title">⬇ Downloads Queue</span>' +
        '<span class="queue-count">' + active.length + ' active</span>' +
      '</div>' +
      '<div class="queue-list">' +
        active.map(function (dl) {
          return self.renderQueueItem(dl);
        }).join('') +
      '</div>';

    // Wire cancel buttons via TRUE event delegation on the stable container.
    // Rows redrawn in place (renderQueueItemInPlace → existing.outerHTML) mint
    // brand-new button nodes; a per-button listener would not survive that
    // (e.g. cancelDownload's revert() redraws an enabled, listener-less button
    // that would then be a silent no-op). Bind once to the container — which
    // innerHTML/outerHTML on its children never replaces — and match on click.
    if (!container._cancelDelegated) {
      container._cancelDelegated = true;
      container.addEventListener('click', function (e) {
        var btn = e.target.closest && e.target.closest('.queue-item-cancel');
        if (!btn || !container.contains(btn)) return;
        e.stopPropagation();
        var pauseJob = btn.getAttribute('data-pause-job');
        if (pauseJob) {
          btn.disabled = true;
          btn.textContent = '…';
          self.pauseDownload(pauseJob);
          return;
        }
        var resumeRepo = btn.getAttribute('data-resume-repo');
        if (resumeRepo) {
          btn.disabled = true;
          btn.textContent = '…';
          self.resumeDownload(resumeRepo);
          return;
        }
        var jobId = btn.getAttribute('data-job-id');
        if (!jobId) return;
        btn.disabled = true;
        btn.textContent = '…';
        self.cancelDownload(jobId);
      });
    }
  },

  // Pause keeps the partial files; cancel deletes them. That is the whole
  // difference, and it is worth a separate button on a 200 GB pull.
  pauseDownload(jobId) {
    var self = this;
    fetch('/api/models/download-pause', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId }),
    }).then(function (r) { return r.json().catch(function () { return {}; }); })
      .then(function (data) {
        Object.keys(self.state.activeDownloads || {}).forEach(function (repo) {
          var dl = self.state.activeDownloads[repo];
          if (dl.jobId === jobId) {
            dl.status = data.status === 'pausing' ? 'pausing' : dl.status;
            self.renderQueueItemInPlace(repo);
          }
        });
        if (data.error) self.toast(data.error, 'error');
      }).catch(function (err) { self.toast('Error: ' + err.message, 'error'); });
  },

  // What the resume verified before it started. Worth saying: the point of
  // the button is that it checks, and a silent check is indistinguishable from
  // no check at all.
  resumeCheckNote(checked) {
    return AINodeLib.resumeCheckNote(checked);
  },

  resumeDownload(hfRepo) {
    var self = this;
    fetch('/api/models/download-resume', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hf_repo: hfRepo }),
    }).then(function (r) { return r.json().catch(function () { return {}; }); })
      .then(function (data) {
        if (data.error) { self.toast(data.error, 'error'); return; }
        self.state.activeDownloads = self.state.activeDownloads || {};
        var previous = self.state.activeDownloads[hfRepo] || {};
        self.state.activeDownloads[hfRepo] = {
          jobId: data.job_id, hfRepo: hfRepo, status: 'downloading',
          startedAt: Date.now(), elapsed: 0,
          totalBytes: previous.totalBytes || 0,
          downloadedBytes: previous.downloadedBytes || 0,
          progress: previous.progress,
        };
        self.saveActiveDownloads();
        self.renderDownloadsQueue();
        self.resumeDownloadPolling(hfRepo);
        self.toast('Resuming ' + hfRepo + self.resumeCheckNote(data.checked),
                   'info');
      }).catch(function (err) { self.toast('Error: ' + err.message, 'error'); });
  },

  cancelDownload(jobId) {
    var self = this;
    // Revert any rows for this job back to 'downloading' so the poll resumes
    // and the cancel button (currently disabled/…) is redrawn fresh.
    var revert = function () {
      Object.keys(self.state.activeDownloads || {}).forEach(function (repo) {
        var dl = self.state.activeDownloads[repo];
        if (dl.jobId === jobId) {
          dl.status = 'downloading';
          self.renderQueueItemInPlace(repo);
        }
      });
    };
    fetch('/api/models/download-cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId }),
    })
    .then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        if (!r.ok) {
          throw new Error((data.error && data.error.message) || data.error || ('HTTP ' + r.status));
        }
        return data;
      });
    })
    .then(function (data) {
      if (data.status === 'cancelling') {
        // Mark the local state so the UI updates immediately
        Object.keys(self.state.activeDownloads || {}).forEach(function (repo) {
          var dl = self.state.activeDownloads[repo];
          if (dl.jobId === jobId) {
            dl.status = 'cancelling';
            self.renderQueueItemInPlace(repo);
          }
        });
      } else {
        // Unexpected response shape — don't leave the row stuck.
        revert();
      }
    })
    .catch(function (err) {
      self.toast('Cancel failed: ' + err.message, 'error');
      revert();
    });
  },

  renderQueueItem(dl) {
    var mins = Math.floor(dl.elapsed / 60);
    var secs = dl.elapsed % 60;
    var elapsedStr = mins > 0 ? mins + 'm ' + secs + 's' : secs + 's';

    var statusClass = dl.status === 'completed' ? 'done' :
                      dl.status === 'failed' ? 'failed' :
                      'active';

    var pct = dl.progress;
    var hasPct = pct != null && !isNaN(pct);
    var pctStr = hasPct ? pct.toFixed(1) + '%' : '';

    var total = dl.totalBytes || 0;
    var got = dl.downloadedBytes || 0;
    var sizeStr = '';
    if (total > 0) {
      sizeStr = this.formatBytes(got) + ' / ' + this.formatBytes(total);
    } else if (got > 0) {
      sizeStr = this.formatBytes(got) + ' downloaded';
    }

    // Throughput + ETA
    var rate = dl.elapsed > 0 ? got / dl.elapsed : 0;
    var rateStr = rate > 0 ? this.formatBytes(rate) + '/s' : '';
    var etaStr = '';
    if (rate > 0 && total > 0 && got < total) {
      var remaining = (total - got) / rate;
      etaStr = 'ETA ' + this.formatDuration(remaining);
    }

    var statusLabel;
    var barHtml = '';
    var cancelBtn = '';
    if (dl.status === 'completed') {
      statusLabel = '✓ Complete · ' + (total > 0 ? this.formatBytes(total) : elapsedStr);
    } else if (dl.status === 'failed') {
      statusLabel = '⚠ Failed' + (dl.error ? ' — ' + this.esc(dl.error) : '');
    } else if (dl.status === 'cancelled') {
      statusLabel = '✕ Cancelled';
    } else if (dl.status === 'cancelling') {
      statusLabel = 'Cancelling...';
    } else if (dl.status === 'pausing') {
      statusLabel = 'Pausing — finishing the file in flight...';
    } else if (dl.status === 'paused') {
      statusLabel = '⏸ Paused' + (sizeStr ? ' · ' + sizeStr : '') +
        ' — Resume continues from here';
      cancelBtn = '<button class="queue-item-cancel" data-resume-repo="' +
        this.esc(dl.hfRepo) + '" title="Resume download">▶</button>';
    } else if (hasPct) {
      statusLabel = pctStr + ' · ' + sizeStr + (rateStr ? ' · ' + rateStr : '') + (etaStr ? ' · ' + etaStr : '');
      barHtml =
        '<div class="queue-item-bar">' +
          '<div class="queue-item-bar-progress" style="width:' + pct.toFixed(2) + '%"></div>' +
        '</div>';
      cancelBtn = '<button class="queue-item-cancel" data-pause-job="' + this.esc(dl.jobId) + '" title="Pause — keeps what has been downloaded">⏸</button>' + 
        '<button class="queue-item-cancel" data-job-id="' + this.esc(dl.jobId) + '" title="Cancel download — deletes what has been downloaded">✕</button>';
    } else {
      // No total yet — show indeterminate shimmer
      statusLabel = 'Starting... ' + elapsedStr + (sizeStr ? ' · ' + sizeStr : '');
      barHtml =
        '<div class="queue-item-bar">' +
          '<div class="queue-item-bar-fill"></div>' +
        '</div>';
      cancelBtn = '<button class="queue-item-cancel" data-pause-job="' + this.esc(dl.jobId) + '" title="Pause — keeps what has been downloaded">⏸</button>' + 
        '<button class="queue-item-cancel" data-job-id="' + this.esc(dl.jobId) + '" title="Cancel download — deletes what has been downloaded">✕</button>';
    }

    return '<div class="queue-item ' + statusClass + '" data-queue-repo="' + this.esc(dl.hfRepo) + '">' +
      '<div class="queue-item-info">' +
        '<div class="queue-item-repo">' + this.esc(dl.hfRepo) + (cancelBtn ? ' ' + cancelBtn : '') + '</div>' +
        '<div class="queue-item-status">' + statusLabel + '</div>' +
      '</div>' +
      barHtml +
    '</div>';
  },

  formatBytes(bytes) {
    return AINodeLib.formatBytes(bytes);
  },

  formatDuration(seconds) {
    if (!seconds || seconds < 0 || !isFinite(seconds)) return '—';
    seconds = Math.round(seconds);
    if (seconds < 60) return seconds + 's';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm ' + (seconds % 60) + 's';
    var h = Math.floor(seconds / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    return h + 'h ' + m + 'm';
  },

  updateNavDownloadBadge() {
    var navPill = document.querySelector('.nav-pill[data-view="downloads"]');
    if (!navPill) return;
    var active = Object.values(this.state.activeDownloads || {}).filter(function (dl) {
      return dl.status === 'downloading';
    });
    var existingBadge = navPill.querySelector('.nav-pill-badge');
    if (active.length === 0) {
      if (existingBadge) existingBadge.remove();
      return;
    }
    if (existingBadge) {
      existingBadge.textContent = active.length;
    } else {
      var badge = document.createElement('span');
      badge.className = 'nav-pill-badge';
      badge.textContent = active.length;
      navPill.appendChild(badge);
    }
  },

  updateDownloadProgress(hfRepo) {
    var dl = (this.state.activeDownloads || {})[hfRepo];
    if (!dl) return;
    // Find all download cards for this model (regardless of which tab user is on)
    document.querySelectorAll('[data-model-id="' + CSS.escape(hfRepo) + '"]').forEach(function (card) {
      var actions = card.querySelector('.download-card-actions');
      if (!actions) return;
      var mins = Math.floor(dl.elapsed / 60);
      var secs = dl.elapsed % 60;
      var timeStr = mins > 0 ? mins + 'm ' + secs + 's' : secs + 's';
      var label = dl.status === 'failed' ? '⚠ Failed' :
                  dl.status === 'completed' ? '✓ Downloaded' :
                  'Downloading ' + timeStr;
      actions.innerHTML =
        '<div class="download-progress-block">' +
          '<div class="download-progress-label">' + label + '</div>' +
          (dl.status === 'downloading' ?
            '<div class="download-progress-bar"><div class="download-progress-fill"></div></div>' : '') +
        '</div>';
    });
  },

  renderCapabilityBadges(model) {
    var caps = model.capabilities || [];
    var defs = {
      vision:       { label: 'Vision',       icon: '👁',  cls: 'cap-vision' },
      tool_use:     { label: 'Tool Use',     icon: '🔧', cls: 'cap-tool' },
      reasoning:    { label: 'Reasoning',    icon: '🧠', cls: 'cap-reasoning' },
      code:         { label: 'Code',         icon: '❮❯', cls: 'cap-code' },
      multilingual: { label: 'Multilingual', icon: '🌐', cls: 'cap-multilingual' },
    };
    var self = this;
    var repo = model.hf_repo || model.id;
    return caps.map(function (c) {
      var d = defs[c];
      if (!d) return '';
      return '<span class="cap-badge ' + d.cls + '" data-info-repo="' + self.esc(repo) + '" title="Click for details">' +
        '<span class="cap-badge-icon">' + d.icon + '</span>' + self.esc(d.label) +
      '</span>';
    }).join('');
  },

  showModelDetail(repoOrId) {
    var self = this;
    // Look up the model from our active source (catalog first, then live-catalog buffer)
    var pool = (this.state.catalog || []).slice();
    if (this.state.liveCatalogBuffer) pool = pool.concat(this.state.liveCatalogBuffer);
    var model = pool.find(function (m) {
      return m.hf_repo === repoOrId || m.id === repoOrId;
    });
    if (!model) {
      // Fetch details from HF search as a fallback
      fetch('/api/models/search?q=' + encodeURIComponent(repoOrId.split('/').pop() || repoOrId) + '&limit=5')
        .then(function (r) { return r.json(); })
        .then(function (data) {
          var hit = (data.models || []).find(function (m) { return m.hf_repo === repoOrId; });
          if (hit) self.renderModelDetailModal(hit);
          else self.toast('Model details unavailable', 'error');
        }).catch(function () { self.toast('Failed to fetch details', 'error'); });
      return;
    }
    this.renderModelDetailModal(model);
  },

  renderModelDetailModal(m) {
    var self = this;
    var existing = document.getElementById('model-detail-modal');
    if (existing) existing.remove();

    var repo = m.hf_repo || m.id;
    var name = m.name || repo;
    var isLoaded = ((this.state.status && this.state.status.models_loaded) || []).includes(repo);
    var isDownloaded = isLoaded || !!(this.state.downloadedModels && this.state.downloadedModels[repo]);
    var capDefs = {
      vision:       { label: 'Vision',       icon: '👁',  cls: 'cap-vision' },
      tool_use:     { label: 'Tool Use',     icon: '🔧', cls: 'cap-tool' },
      reasoning:    { label: 'Reasoning',    icon: '🧠', cls: 'cap-reasoning' },
      code:         { label: 'Code',         icon: '❮❯', cls: 'cap-code' },
      multilingual: { label: 'Multilingual', icon: '🌐', cls: 'cap-multilingual' },
    };
    var caps = (m.capabilities || []).map(function (c) {
      var d = capDefs[c]; if (!d) return '';
      return '<span class="cap-badge large ' + d.cls + '"><span class="cap-badge-icon">' + d.icon + '</span>' + d.label + '</span>';
    }).join('');

    var sizeStr = m.size_gb ? '~' + Math.round(m.size_gb) + ' GB' : (m.size || 'size unknown');
    var downloadsStr = m.downloads ? self.formatNumber(m.downloads) : '—';
    var likesStr = m.likes ? self.formatNumber(m.likes) : '—';
    var ageStr = m.created_at ? self.relativeTime(m.created_at).replace('released ', '') : '—';
    var license = m.license || '—';
    var arch = (m.architecture || m.family || '—');
    var fmt = m.format || (m.quantization ? m.quantization.toUpperCase() : 'SafeTensors');
    var paramsStr = m.params_b ? m.params_b + 'B' : (m.params || '—');
    var ctxStr = m.context_length ? self.formatNumber(m.context_length) + ' tokens' : '—';

    var metaRow =
      '<div class="md-meta-row">' +
        '<div class="md-meta-pair"><span class="md-meta-label">Params</span><span class="md-meta-value">' + self.esc(paramsStr) + '</span></div>' +
        '<div class="md-meta-pair"><span class="md-meta-label">Arch</span><span class="md-meta-value">' + self.esc(arch) + '</span></div>' +
        '<div class="md-meta-pair"><span class="md-meta-label">Format</span><span class="md-meta-value md-format">' + self.esc(fmt) + '</span></div>' +
        '<div class="md-meta-pair"><span class="md-meta-label">Context</span><span class="md-meta-value">' + self.esc(ctxStr) + '</span></div>' +
        '<div class="md-meta-pair"><span class="md-meta-label">License</span><span class="md-meta-value">' + self.esc(license) + '</span></div>' +
      '</div>';

    var primaryCta = isLoaded
      ? '<button class="btn-nvidia md-cta" id="md-use-chat">▶ Use in New Chat</button>'
      : isDownloaded
        ? '<button class="btn-nvidia md-cta" id="md-launch" data-hf-repo="' + self.esc(repo) + '">▶ Launch Model</button>'
        : '<button class="btn-nvidia md-cta" id="md-download" data-hf-repo="' + self.esc(repo) + '">▼ Download (' + sizeStr + ')</button>';

    var modal = document.createElement('div');
    modal.id = 'model-detail-modal';
    modal.className = 'model-detail-modal-overlay';
    modal.innerHTML =
      '<div class="model-detail-modal">' +
        '<div class="md-header">' +
          '<div class="md-header-left">' +
            '<div class="md-icon">⬡</div>' +
            '<div class="md-title">' + self.esc(repo) + '</div>' +
            '<button class="md-copy" data-copy="' + self.esc(repo) + '" title="Copy repo">⧉</button>' +
          '</div>' +
          '<button class="md-close" title="Close">×</button>' +
        '</div>' +
        '<div class="md-stats-row">' +
          '<div class="md-stat"><span class="md-stat-icon">⬇</span>' + downloadsStr + '</div>' +
          '<div class="md-stat"><span class="md-stat-icon">★</span>' + likesStr + '</div>' +
          '<div class="md-stat md-stat-age">Last updated: ' + self.esc(ageStr) + '</div>' +
          (m.recommended ? '<div class="md-staff-pick">✨ Recommended</div>' : '') +
        '</div>' +
        (m.description || m.desc ? '<div class="md-description">' + self.esc(m.description || m.desc) + '</div>' : '') +
        metaRow +
        (caps ? '<div class="md-capabilities"><span class="md-section-label">Capabilities</span><div class="md-cap-list">' + caps + '</div></div>' : '') +
        '<div class="md-footer">' +
          '<div class="md-footer-status">' + (isLoaded ? '● Model loaded and ready' : isDownloaded ? '◉ Downloaded — click Launch to run' : '○ Not yet downloaded') + '</div>' +
          primaryCta +
        '</div>' +
      '</div>';

    document.body.appendChild(modal);

    // Close handlers
    var close = function () { modal.remove(); };
    modal.querySelector('.md-close').addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
    document.addEventListener('keydown', function escHandler(e) {
      if (e.key === 'Escape') { close(); document.removeEventListener('keydown', escHandler); }
    });

    // Copy repo
    modal.querySelector('.md-copy').addEventListener('click', function (e) {
      navigator.clipboard.writeText(e.currentTarget.dataset.copy);
      self.toast('Repo copied', 'success');
    });

    // Download
    var dlBtn = modal.querySelector('#md-download');
    if (dlBtn) {
      dlBtn.addEventListener('click', function () {
        self.startRepoDownload(dlBtn.dataset.hfRepo);
        close();
      });
    }

    // Launch downloaded model (set as active model + restart engine)
    var launchBtn = modal.querySelector('#md-launch');
    if (launchBtn) {
      launchBtn.addEventListener('click', function () {
        launchBtn.disabled = true;
        launchBtn.textContent = 'Launching...';
        fetch('/api/engine/set-model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: repo }),
        })
        .then(function (r) { return r.json(); })
        .then(function () {
          self.toast('Launching ' + repo + ' — engine restarting', 'success');
          close();
        })
        .catch(function () {
          self.toast('Failed to launch model', 'error');
          launchBtn.disabled = false;
          launchBtn.textContent = '▶ Launch Model';
        });
      });
    }

    // Use in chat
    var chatBtn = modal.querySelector('#md-use-chat');
    if (chatBtn) {
      chatBtn.addEventListener('click', function () {
        self.state.messages = [];
        self.state.currentConversation = null;
        self.navigate('chat');
        var sel = document.getElementById('chat-model');
        if (sel) {
          for (var i = 0; i < sel.options.length; i++) {
            if (sel.options[i].value === repo) { sel.value = repo; break; }
          }
        }
        close();
      });
    }
  },

  relativeTime(isoString) {
    if (!isoString) return '';
    try {
      var then = new Date(isoString);
      var now = new Date();
      var diffMs = now - then;
      if (isNaN(diffMs) || diffMs < 0) return '';
      var seconds = Math.floor(diffMs / 1000);
      var minutes = Math.floor(seconds / 60);
      var hours = Math.floor(minutes / 60);
      var days = Math.floor(hours / 24);
      var months = Math.floor(days / 30);
      var years = Math.floor(days / 365);
      if (years > 0) return 'released ' + years + 'y ago';
      if (months > 0) return 'released ' + months + 'mo ago';
      if (days > 0) return 'released ' + days + 'd ago';
      if (hours > 0) return 'released ' + hours + 'h ago';
      if (minutes > 0) return 'released ' + minutes + 'm ago';
      return 'just now';
    } catch (e) {
      return '';
    }
  },

  renderHuggingFaceSearch(container, loaded, gpuMem, clusterNodeCount, totalClusterMem, filterPills, totalCount) {
    var self = this;
    var query = this.state.hfSearchQuery || '';
    // Always rebuild if we're entering HF mode from catalog mode
    var hfInput = container.querySelector('#hf-search-input');
    if (!hfInput) {
      var toolbarHtml = '<div class="downloads-header">' +
        '<h2 class="view-title">🤗 Hugging Face Hub</h2>' +
        '<div class="downloads-count" id="hf-count">Search millions of models</div>' +
        '</div>' +
        '<div id="downloads-queue" class="downloads-queue"></div>' +
        '<div class="downloads-toolbar">' +
        '<input type="text" id="hf-search-input" class="search-input" placeholder="Search HuggingFace (e.g. llama, qwen, mistral, phi)..." value="' + this.esc(query) + '" autofocus>' +
        '<button id="hf-back-btn" class="btn-sm hf-back-btn">← Back to Catalog</button>' +
        '</div>' +
        // The kinds decide which pipeline tags are queried, so this is part
        // of the search and not a filter applied to its results.
        '<div class="pill-group hf-kind-filter" id="hf-kind-filter"></div>' +
        '<div id="downloads-results"></div>';
      container.innerHTML = toolbarHtml;
    }

    var resultsContainer = container.querySelector('#downloads-results');
    var countEl = container.querySelector('#hf-count');
    this.renderHfFilters();

    // Back to the two-list catalog.
    var backBtn = container.querySelector('#hf-back-btn');
    if (backBtn && !backBtn.dataset.bound) {
      backBtn.dataset.bound = '1';
      backBtn.addEventListener('click', function () {
        self.state.modelsSearch = '';
        self.state.modelsFilter = 'catalog';
        self.renderDownloads();
      });
    }

    // Bind search input (debounced)
    var input = container.querySelector('#hf-search-input');
    if (input && !input.dataset.bound) {
      input.dataset.bound = '1';
      input.addEventListener('input', function () {
        self.state.hfSearchQuery = input.value;
        clearTimeout(self._hfSearchTimer);
        self._hfSearchTimer = setTimeout(function () {
          self.performHuggingFaceSearch(input.value, resultsContainer, countEl, loaded, gpuMem);
        }, 400);
      });
    }

    // Initial: if we have a query, rerun search; else show hint
    if (query) {
      this.performHuggingFaceSearch(query, resultsContainer, countEl, loaded, gpuMem);
    } else if (resultsContainer && !resultsContainer.innerHTML) {
      resultsContainer.innerHTML = '<div class="downloads-empty">Type a search query to find models on HuggingFace Hub.</div>';
    }
  },

  performHuggingFaceSearch(query, resultsContainer, countEl, loaded, gpuMem) {
    var self = this;
    if (!query || query.length < 2) {
      if (resultsContainer) resultsContainer.innerHTML = '<div class="downloads-empty">Type at least 2 characters to search.</div>';
      if (countEl) countEl.textContent = 'Search for any text-generation model';
      return;
    }

    if (resultsContainer) resultsContainer.innerHTML = '<div class="downloads-empty">Searching HuggingFace...</div>';

    var kinds = this.state.hfKinds || [];
    fetch('/api/models/search?q=' + encodeURIComponent(query) + '&limit=50' +
          (kinds.length ? '&kind=' + encodeURIComponent(kinds.join(',')) : ''))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        self._hfResults = data.models || [];
        self._hfQuery = query;
        self.state.hfShowAll = false;  // reset the reveal on each new search
        self.renderHfResults();
      })
      .catch(function (err) {
        var rc = document.getElementById('downloads-results');
        if (rc) rc.innerHTML = '<div class="downloads-empty">Search failed: ' + err.message + '</div>';
      });
  },

  // Render cached HF results, hiding models that can't run here (incompatible
  // engine or too large) unless "Show all" is toggled (then shown dimmed).
  renderHfResults() {
    var self = this;
    var models = this._hfResults || [];
    var query = this._hfQuery || '';
    var loaded = (this.state.status && this.state.status.models_loaded) || [];
    var resultsContainer = document.getElementById('downloads-results');
    var countEl = document.getElementById('hf-count');
    if (!resultsContainer) return;

    var runnable = models.filter(function (m) { return self.hfRunnable(m); });
    var hiddenCount = models.length - runnable.length;
    var showAll = !!this.state.hfShowAll;
    var shown = showAll ? models : runnable;

    this.renderHfFilters();
    if (countEl) {
      var kinds = this.state.hfKinds || [];
      countEl.textContent = runnable.length + ' runnable for "' + query + '"' +
        (kinds.length ? ' in ' + kinds.join(' + ') : '') +
        (hiddenCount ? ' · ' + hiddenCount + ' this cluster cannot load' : '');
    }

    function card(m) {
      var isLoaded = loaded.includes(m.hf_repo);
      var isOnDisk = isLoaded || !!(self.state.downloadedModels && self.state.downloadedModels[m.hf_repo]);
      var dimmed = !self.hfRunnable(m);
      var sizeStr = m.size_gb > 0 ? '~' + Math.round(m.size_gb) + ' GB' : 'size unknown';
      var fitBadge = self.placementBadge(m);
      var catalogBadge = m.in_catalog ? '<span class="fit-badge rec">✓ Recommended</span>' : '';
      var quantBadge = m.quant ? '<span class="fit-badge ' + (/MLX|GGUF/.test(m.quant) ? 'untested' : 'quant') + '">' + self.esc(m.quant) + '</span>' : '';
      var kindBadge = {
        image: '<span class="fit-badge">\u25a3 Image generation</span>',
        vision: '<span class="fit-badge">\u25c9 Vision</span>',
        embedding: '<span class="fit-badge">\u2261 Embeddings</span>',
      }[m.kind] || '';
      // Why it cannot run here, in words. "Dimmed" tells someone that
      // something is wrong and nothing about what — and the answer is
      // usually one sentence long.
      var whyNot = (!self.hfRunnable(m) && m.not_servable_reason)
        ? '<div class="hf-why-not">' + self.esc(m.not_servable_reason) + '</div>'
        : '';
      var statusBadge = isLoaded ? '<span class="model-badge loaded">Loaded</span>'
        : isOnDisk ? '<span class="model-badge loaded">Downloaded</span>'
        : '<span class="model-badge available">Available</span>';
      var downloadsStr = m.downloads ? self.formatNumber(m.downloads) + ' downloads' : '';
      var detailsBtn = '<button class="btn-sm download-details-btn" data-info-repo="' + self.esc(m.hf_repo) + '">Details</button>';
      var downloadBtn = isOnDisk ? '' : '<button class="btn-sm downloads-download-btn" data-model-id="' + self.esc(m.hf_repo) + '">Download</button>';
      return '<div class="download-card' + (dimmed ? ' dimmed' : '') + '" data-model-id="' + self.esc(m.hf_repo) + '">' +
        '<div class="download-card-main">' +
        '<div class="download-card-info">' +
        '<div class="download-card-header">' +
        '<div class="download-card-name">' + self.esc(m.name) + '</div>' +
        '<div class="download-card-badges">' + catalogBadge + kindBadge + quantBadge + fitBadge + statusBadge + '</div>' +
        '</div>' +
        '<div class="download-card-repo">' + self.esc(m.hf_repo) + '</div>' +
        '<div class="download-card-desc">' + sizeStr + (downloadsStr ? ' &middot; ' + downloadsStr : '') + '</div>' +
        whyNot +
        '</div>' +
        '<div class="download-card-actions">' + detailsBtn + downloadBtn + '</div>' +
        '</div>' +
        '</div>';
    }

    var body = shown.length
      ? '<div class="downloads-grid">' + shown.map(card).join('') + '</div>'
      : '<div class="downloads-empty">No models here can run on this cluster.</div>';
    var toggle = hiddenCount ? '<div class="hf-toggle-row"><button id="hf-show-all" class="btn-sm">' +
      (showAll ? 'Hide incompatible / too-large' : 'Show ' + hiddenCount + ' hidden (incompatible or too large)') +
      '</button></div>' : '';
    resultsContainer.innerHTML = body + toggle;
    self.bindRepoDownloadButtons(resultsContainer);
    var tbtn = document.getElementById('hf-show-all');
    if (tbtn) tbtn.addEventListener('click', function () {
      self.state.hfShowAll = !self.state.hfShowAll;
      self.renderHfResults();
    });
  },

  formatNumber(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return String(n);
  },

  // Placement of a model on THIS cluster. Returns {known, tooLarge, tp, label, cls}.
  // Accepts catalog-shaped (sizeGb/minMem/proven_tp) or HF-shaped (size_gb) models.
  placementInfo(model) {
    var nodes = this.state.nodes || [];
    var nodeCount = nodes.length || 1;
    var st = this.state.status;
    var perNode = (st && st.gpu && st.gpu.memory_total_mb) ? st.gpu.memory_total_mb / 1024
      : ((nodes[0] && nodes[0].gpu_memory_gb) || 0);
    var size = model.sizeGb || model.size_gb || 0;
    if (!size || !perNode) return { known: false };  // unknown size → make no claim
    // ponytail: weights × 1.2 for KV/activation headroom; minMem wins if curated.
    var req = (model.minMem && model.minMem > size) ? model.minMem : size * 1.2;
    var needed = Math.ceil(req / perNode);
    var neededTp = needed <= 1 ? 1 : needed <= 2 ? 2 : needed <= 4 ? 4 : 8;
    // Honor the proven launch config when it's at least what memory requires.
    var tp = (model.proven_tp && model.proven_tp >= neededTp) ? model.proven_tp : neededTp;
    if (needed > nodeCount || tp > nodeCount) {
      return { known: true, tooLarge: true, label: 'Too large for cluster', cls: 'no-fit' };
    }
    if (tp === 1) return { known: true, tooLarge: false, tp: 1, label: 'Runs on 1 node', cls: 'fits' };
    return { known: true, tooLarge: false, tp: tp, label: 'Good for TP=' + tp + ' · ' + tp + ' nodes', cls: 'cluster' };
  },

  placementBadge(model) {
    var p = this.placementInfo(model);
    return p.known ? '<span class="fit-badge ' + p.cls + '">' + p.label + '</span>' : '';
  },

  // Can this deployment actually serve this repo? The verdict is the
  // server's — it knows which engine each kind needs and therefore which
  // formats can work — with the name check kept as the fallback for an older
  // node that does not send one.
  hfRunnable(m) {
    var repo = (m.hf_repo || '').toLowerCase();
    var ok = m.servable !== undefined
      ? m.servable !== false
      : (m.vllm_ok !== false && !/mlx|gguf|ggml/.test(repo));
    var info = this.placementInfo(m);
    // Unknown size (GGUF etc.) can't be placed — treat as not-runnable-here.
    return ok && info.known && !info.tooLarge;
  },

  // The kinds this deployment serves, in the order they are offered.
  hfKindOptions() {
    return [
      { id: 'chat', label: 'Chat' },
      { id: 'vision', label: 'Vision' },
      { id: 'image', label: 'Image generation' },
      { id: 'embedding', label: 'Embeddings' },
    ];
  },

  toggleHfKind(kind) {
    var current = this.state.hfKinds || [];
    this.state.hfKinds = current.indexOf(kind) === -1
      ? current.concat([kind])
      : current.filter(function (k) { return k !== kind; });
    // Re-run rather than filter locally: the kinds decide which pipeline tags
    // are queried, so a filter applied after the fact would only narrow the
    // results of the wrong search.
    if (this._hfQuery) this.searchHuggingFace(this._hfQuery);
    else this.renderHfFilters();
  },

  renderHfFilters() {
    var mount = document.getElementById('hf-kind-filter');
    if (!mount) return;
    var self = this;
    var active = this.state.hfKinds || [];
    mount.innerHTML = this.hfKindOptions().map(function (option) {
      var on = active.indexOf(option.id) !== -1;
      return '<button class="pill' + (on ? ' active' : '') +
        '" data-hf-kind="' + option.id + '">' + self.esc(option.label) +
        '</button>';
    }).join('') +
      (active.length
        ? '<button class="pill" data-hf-kind="">All</button>'
        : '<span class="config-field-hint" style="align-self:center">' +
          'everything this cluster can serve</span>');
    mount.querySelectorAll('[data-hf-kind]').forEach(function (button) {
      button.addEventListener('click', function () {
        var kind = button.getAttribute('data-hf-kind');
        if (!kind) {
          self.state.hfKinds = [];
          if (self._hfQuery) self.searchHuggingFace(self._hfQuery);
          else self.renderHfFilters();
          return;
        }
        self.toggleHfKind(kind);
      });
    });
  },

  // ----- Import a model from files -----------------------------------------
  //
  // For a link that cannot carry 86 GB in one piece. The panel says which
  // files are missing and where to get them; the browser then hands them
  // over one at a time, and the head mirrors the result to the peers the
  // same way a download does.

  toggleImportPanel() {
    var panel = document.getElementById('import-panel');
    if (!panel) return;
    if (panel.style.display !== 'none') { panel.style.display = 'none'; return; }
    panel.style.display = '';
    this.renderImportPanel();
    this.loadDropBox();
  },

  renderImportPanel(plan) {
    var panel = document.getElementById('import-panel');
    if (!panel) return;
    var self = this;
    var repo = this.state.importRepo || '';
    var drop = this.state.importDrop;
    var html = '<div class="queue-header"><span class="queue-title">' +
      '📥 Import a model from files</span></div>' +
      this.renderDropBox(drop) +
      '<div style="padding:10px 14px">' +
      '<p class="config-card-desc">Download the files on any machine with a ' +
      'working connection, then hand them over here. Nothing is fetched from ' +
      'Hugging Face by this node, and the head mirrors the finished model to ' +
      'the other nodes.</p>' +
      '<div class="config-form-grid"><div>' +
      '<label class="config-field-label">Repository</label>' +
      '<input class="form-input" id="import-repo" placeholder="org/name" ' +
      'value="' + this.esc(repo) + '"></div></div>' +
      '<div class="config-actions">' +
      '<button class="config-btn" id="import-check">What is missing?</button>' +
      '</div>';

    if (plan) {
      var missing = plan.missing || [];
      html += '<div class="plan-notes"><div>Target: <code>' +
        this.esc(plan.target_dir) + '</code></div>';
      if (plan.source === 'local') {
        html += '<div>⚠ The Hub could not be reached, so this list comes from ' +
          'the checkpoint already on disk. It can name the weight shards and ' +
          'the usual small files, and nothing this repo has beyond them.</div>';
      }
      html += '<div>' + (plan.files || []).length + ' file(s), ' +
        missing.length + ' missing' +
        (plan.missing_bytes ? ' (' + this.formatBytes(plan.missing_bytes) + ')' : '') +
        '</div>';
      if (plan.incomplete_reason) {
        html += '<div>⚠ ' + this.esc(plan.incomplete_reason) + '</div>';
      }
      html += '</div>';

      if (missing.length) {
        html += '<div class="queue-list">' + missing.slice(0, 200).map(function (f) {
          return '<div class="queue-item"><div class="queue-item-info">' +
            '<div class="queue-item-repo"><a href="' + self.esc(f.url) +
            '" target="_blank" rel="noopener">' + self.esc(f.path) + '</a></div>' +
            '<div class="queue-item-status">' +
            (f.size ? self.formatBytes(f.size) : 'size unknown') +
            '</div></div></div>';
        }).join('') + '</div>';
        html += '<div class="config-actions">' +
          '<input type="file" id="import-files" multiple style="display:none">' +
          '<button class="config-btn" id="import-pick">Choose the files…</button>' +
          '<span class="config-field-hint" id="import-progress"></span>' +
          '</div>';
      } else {
        html += '<div class="plan-notes"><div>Every file is here.</div></div>' +
          '<div class="config-actions">' +
          '<button class="config-btn" id="import-finish">Check and distribute</button>' +
          '<span class="config-field-hint" id="import-progress"></span></div>';
      }
    }
    html += '</div>';
    panel.innerHTML = html;

    var check = document.getElementById('import-check');
    if (check) check.addEventListener('click', function () {
      self.state.importRepo =
        (document.getElementById('import-repo') || {}).value || '';
      self.loadImportPlan();
    });
    var pick = document.getElementById('import-pick');
    var input = document.getElementById('import-files');
    if (pick && input) {
      pick.addEventListener('click', function () { input.click(); });
      input.addEventListener('change', function () {
        self.uploadImportFiles(Array.prototype.slice.call(input.files || []));
      });
    }
    var finish = document.getElementById('import-finish');
    if (finish) finish.addEventListener('click', function () { self.finishImport(); });
    panel.querySelectorAll('[data-take]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        self.takeFromDropBox(btn.getAttribute('data-take'), '');
      });
    });
    var loose = panel.querySelector('[data-take-loose]');
    if (loose) loose.addEventListener('click', function () {
      var repo = (document.getElementById('import-repo') || {}).value || '';
      if (!repo || repo.indexOf('/') < 0) {
        self.toast('Name the repository first, as org/name', 'error');
        return;
      }
      self.takeFromDropBox('', repo);
    });
  },

  // The drop directory. A browser upload is right for twenty small files and
  // wrong for 43 GB of shards; a directory and a file manager are right for
  // that, and every operating system ships both.
  renderDropBox(drop) {
    if (!drop) return '';
    var self = this;
    var html = '<div style="padding:10px 14px 0">' +
      '<div class="config-card-desc">Drop directory: <code>' +
      this.esc(drop.dir) + '</code>';
    if (!drop.exists) {
      html += ' — not visible from inside AINode.</div>' +
        '<div class="plan-warn">' + this.esc(drop.hint || '') + '</div></div>';
      return html;
    }
    var entries = drop.entries || [];
    var loose = drop.loose || [];
    if (!entries.length && !loose.length) {
      return html + ' — empty. Put a model there as <code>org--name/</code> ' +
        'and it appears here.</div></div>';
    }
    html += '</div><div class="queue-list">';
    entries.forEach(function (e) {
      html += '<div class="queue-item"><div class="queue-item-info">' +
        '<div class="queue-item-repo">' + self.esc(e.repo || e.name) + '</div>' +
        '<div class="queue-item-status">' + e.files + ' file(s), ' +
        self.formatBytes(e.bytes) + ' — in ' + self.esc(e.name) + '</div></div>' +
        '<button class="config-btn" data-take="' + self.esc(e.name) +
        '">Take it in</button></div>';
    });
    if (loose.length) {
      html += '<div class="queue-item"><div class="queue-item-info">' +
        '<div class="queue-item-repo">' + loose.length +
        ' loose file(s) at the top level</div>' +
        '<div class="queue-item-status">Name the repository above, then take ' +
        'them in — they cannot say which model they belong to.</div></div>' +
        '<button class="config-btn" data-take-loose="1">Take them in</button>' +
        '</div>';
    }
    return html + '</div>';
  },

  async loadDropBox() {
    this.state.importDrop = await this.fetchJSON('/api/models/import/dropbox');
    this.renderImportPanel(this.state.importPlan);
  },

  // "42 stale staging files, 61.4 GB reclaimed" — worth saying out loud,
  // because that space was being held by a transfer the import replaced.
  clearedNote(out) {
    return AINodeLib.clearedNote(out);
  },

  // The mirror is a job, not a reply. A 129 GB push to the peers used to be
  // awaited inside the request: the page (or a curl) waited for the last byte
  // and killing the client killed the transfer. Now the request returns a
  // job_id and this follows it, the same way a download is followed.
  watchImportMirror(jobId, repo) {
    if (!jobId) return;
    var self = this;
    var line = document.getElementById('import-progress');
    var tick = async function () {
      var data = await self.fetchJSON('/api/models/downloads/active')
        .catch(function () { return null; });
      var job = ((data || {}).jobs || []).filter(function (j) {
        return j.job_id === jobId;
      })[0];
      if (!job) { if (line) line.textContent = ''; return; }
      if (job.status === 'importing') {
        if (line) {
          line.textContent = 'moving ' + (repo || job.model_id) + ' in — ' +
            (job.moved || 0) + '/' + (job.files_total == null ? '?' : job.files_total) +
            ' file(s), ' + ((job.moved_bytes || 0) / 1e9).toFixed(1) + ' GB';
        }
        setTimeout(tick, 1000);
        return;
      }
      if (job.status === 'failed' && job.moved != null && job.complete == null) {
        if (line) line.textContent = '';
        self.toast('Could not take ' + (repo || job.model_id) + ' in: ' +
                   (job.error || 'failed') + ' (' + job.moved + ' file(s) moved)', 'error');
        self.invalidate();
        self.loadDropBox().catch(function () {});
        return;
      }
      if (job.complete != null) {
        // Once, when the move is over: what was taken in and whether it is
        // whole. The mirror line takes over from here.
        if (!self.state.importTook) self.state.importTook = {};
        if (!self.state.importTook[jobId]) {
          self.state.importTook[jobId] = true;
          self.toast('Took ' + job.moved + ' file(s) in' + self.clearedNote(job) +
                     (job.complete ? ' — now sending it to the other nodes'
                                   : ' — ' + (job.incomplete_reason || 'still incomplete')),
                     job.complete ? 'success' : 'info');
          self.invalidate();
          self.loadDropBox().then(function () { return self.loadImportPlan(); })
            .catch(function () {});
        }
      }
      if (job.status === 'imported') {
        if (line) line.textContent = '';
        self.renderDownloads();
        return;
      }
      var states = job.mirror || {};
      var names = Object.keys(states);
      var done = names.filter(function (n) { return states[n] === 'ok'; });
      if (job.status === 'mirroring') {
        if (line) {
          line.textContent = 'sending ' + (repo || job.model_id) +
            ' to the other nodes — ' + done.length + '/' +
            (names.length || '?') + ' done' +
            (names.length ? ' (' + names.map(function (n) {
              return n + ': ' + states[n];
            }).join(', ') + ')' : '');
        }
        setTimeout(tick, 2000);
        return;
      }
      if (line) line.textContent = '';
      if (job.status === 'completed') {
        self.toast('Sent ' + (repo || job.model_id) + ' to ' + done.length +
                   ' node(s)', 'success');
      } else {
        self.toast('Could not send ' + (repo || job.model_id) + ' to the ' +
                   'nodes: ' + (job.error || job.status), 'error');
      }
      self.invalidate();
      self.renderDownloads();
    };
    tick();
  },

  async takeFromDropBox(name, repo) {
    var progress = document.getElementById('import-progress');
    if (progress) progress.textContent = 'moving the files in…';
    try {
      var resp = await fetch('/api/models/import/dropbox', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name || undefined, hf_repo: repo || undefined }),
      });
      var out = await resp.json().catch(function () { return {}; });
      if (progress) progress.textContent = '';
      if (!resp.ok || out.error) {
        this.toast(out.error || 'Could not take it in', 'error');
        return;
      }
      // A job now, not a reply: with /model-import on its own disk the move
      // is a copy, and it used to hold the whole server until it finished.
      // The same job goes on to send the model to the other nodes.
      this.state.importRepo = out.hf_repo || this.state.importRepo;
      this.watchImportMirror(out.job_id, out.hf_repo);
    } catch (err) {
      if (progress) progress.textContent = '';
      this.toast('Error: ' + err.message, 'error');
    }
  },

  async loadImportPlan() {
    var repo = (this.state.importRepo || '').trim();
    if (!repo || repo.indexOf('/') < 0) {
      this.toast('Enter a repository as org/name', 'error');
      return;
    }
    var plan = await this.fetchJSON(
      '/api/models/import/plan?hf_repo=' + encodeURIComponent(repo));
    if (!plan || plan.error) {
      this.toast((plan && plan.error) || 'Could not read that repository', 'error');
      return;
    }
    this.state.importPlan = plan;
    this.renderImportPanel(plan);
  },

  async uploadImportFiles(files) {
    if (!files.length) return;
    var repo = (this.state.importRepo || '').trim();
    var progress = document.getElementById('import-progress');
    var plan = this.state.importPlan || {};
    // Match what the browser hands over against what the plan expects, so a
    // file picked from a subdirectory lands in that subdirectory here.
    var byName = {};
    (plan.files || []).forEach(function (f) {
      byName[f.path.split('/').pop()] = f.path;
    });
    for (var i = 0; i < files.length; i++) {
      var file = files[i];
      var relative = byName[file.name] || file.name;
      if (progress) {
        progress.textContent = 'uploading ' + (i + 1) + '/' + files.length +
          ' — ' + file.name;
      }
      var form = new FormData();
      form.append('hf_repo', repo);
      form.append('path', relative);
      form.append('file', file);
      try {
        var resp = await fetch('/api/models/import/upload',
                               { method: 'POST', body: form });
        var out = await resp.json().catch(function () { return {}; });
        if (!resp.ok || out.error) {
          this.toast(out.error || ('Upload failed: ' + file.name), 'error');
          if (progress) progress.textContent = '';
          return;
        }
      } catch (err) {
        this.toast('Upload failed: ' + err.message, 'error');
        if (progress) progress.textContent = '';
        return;
      }
    }
    if (progress) progress.textContent = '';
    this.toast(files.length + ' file(s) imported', 'success');
    this.invalidate();
    await this.loadImportPlan();
  },

  async finishImport() {
    var repo = (this.state.importRepo || '').trim();
    var progress = document.getElementById('import-progress');
    if (progress) progress.textContent = 'checking and distributing…';
    try {
      var resp = await fetch('/api/models/import/finish', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hf_repo: repo }),
      });
      var out = await resp.json().catch(function () { return {}; });
      if (progress) progress.textContent = '';
      if (!resp.ok || out.error) {
        this.toast(out.error || 'Could not finish the import', 'error');
        return;
      }
      if (!out.complete) {
        this.toast((out.incomplete_reason || 'Still incomplete') +
                   this.clearedNote(out), 'error');
      } else {
        this.toast('Imported' + this.clearedNote(out) +
                   ' — now sending it to the other nodes', 'success');
        this.watchImportMirror(out.job_id, out.hf_repo);
      }
      this.invalidate();
      this.renderDownloads();
      await this.loadImportPlan();
    } catch (err) {
      if (progress) progress.textContent = '';
      this.toast('Error: ' + err.message, 'error');
    }
  },

  renderDownloads() {
    var container = document.getElementById('downloads-content');
    if (!container) return;
    var s = this.state.status;
    var loaded = s?.models_loaded || [];
    var self = this;
    var nodes = this.state.nodes;
    var gpuMem = s?.gpu?.memory_total_mb ? s.gpu.memory_total_mb / 1024 : 0;
    var totalClusterMem = nodes.reduce(function (sum, n) { return sum + (n.gpu_memory_gb || 0); }, 0);
    var clusterNodeCount = nodes.length;

    // What is on disk, ACROSS THE CLUSTER and not only here. The page used
    // to scan this node's disk while the panel beside it listed what the
    // whole cluster was serving: a model downloaded to node 3 and running
    // there appeared in one and not the other, which reads as the page having
    // lost it. Refreshed on a TTL, because it used to be fetched exactly once
    // per page load — so anything downloaded or started afterwards stayed
    // invisible until the operator reloaded the browser.
    if (this._stale('downloadedModels')) {
      this.state.downloadedModels = this.state.downloadedModels || {};
      fetch('/api/cluster/models').then(function (r) { return r.json(); })
        .then(function (data) {
          var map = {};
          var list = (data && data.models) || [];
          list.forEach(function (m) {
            var repo = m.hf_repo || m.id || '';
            if (repo) map[repo] = true;
          });
          self.state.downloadedModels = map;
          self.state.downloadedList = list;
          self.state.modelNodeNames = (data && data.node_names) || {};
          self.renderDownloads();
        }).catch(function () {});
    }

    // Which models the memory guard has stopped somewhere. On the same TTL
    // as the rest: a card that says "On disk" about a model every launch
    // refuses is telling the truth and answering the wrong question.
    if (this._stale('blockedModels')) {
      this.state.blockedModels = this.state.blockedModels || {};
      fetch('/api/cluster/measurements').then(function (r) { return r.json(); })
        .then(function (data) {
          var map = {};
          self._blockedModels(data).forEach(function (b) {
            map[b.model] = (map[b.model] || 0) + b.stops;
          });
          self.state.blockedModels = map;
          self.renderDownloads();
        }).catch(function () {});
    }

    // Fetch catalog from API (42+ models) — refreshed on the same TTL
    if (this._stale('catalog')) {
      this.state.catalog = this.state.catalog || [];
      fetch('/api/models').then(function (r) { return r.json(); }).then(function (data) {
        self.state.catalog = (data.models || []).map(function (m) {
          return {
            id: m.hf_repo || m.id,
            slug: m.id,
            name: m.name,
            size: '~' + Math.round(m.size_gb) + ' GB',
            sizeGb: m.size_gb,
            desc: m.description,
            family: m.family || '',
            params: m.params_b ? m.params_b + 'B' : '',
            quantization: m.quantization,
            modality: m.modality || 'text',
            minMem: m.min_memory_gb || m.size_gb,
            recommended: m.recommended || false,
            created_at: m.created_at || '',
            downloads: m.downloads || 0,
            likes: m.likes || 0,
            capabilities: m.capabilities || [],
            architecture: m.architecture || '',
            format: m.format || '',
            hf_repo: m.hf_repo || '',
            context_length: m.context_length || 0,
            license: m.license || '',
            verified: m.verified === true,
            curated: m.curated === true,
            proven_tp: m.proven_tp || 1,
            downloaded: m.downloaded === true,
          };
        });
        self.renderDownloads();
      }).catch(function () { self.state.catalog = []; });
      container.innerHTML = '<div class="downloads-empty">Loading catalog...</div>';
      return;
    }

    var catalog = this.state.catalog;
    var query = (this.state.modelsSearch || '').toLowerCase();
    var dlMap = this.state.downloadedModels || {};

    // Browse HuggingFace — the single escape hatch for grabbing anything not
    // in our known-good catalog.
    if (this.state.modelsFilter === 'huggingface') {
      return this.renderHuggingFaceSearch(container, loaded, gpuMem, clusterNodeCount, totalClusterMem, '', catalog.length);
    }

    function matchesQuery(m) {
      if (!query) return true;
      return (m.id || '').toLowerCase().indexOf(query) !== -1 ||
             (m.name || '').toLowerCase().indexOf(query) !== -1 ||
             (m.desc || '').toLowerCase().indexOf(query) !== -1 ||
             (m.family || '').toLowerCase().indexOf(query) !== -1;
    }
    function isOnDisk(m) {
      return m.downloaded === true || !!dlMap[m.hf_repo] || !!dlMap[m.id] ||
             loaded.includes(m.slug) || loaded.includes(m.id);
    }

    // Two lists, period: what you HAVE (on disk) and the curated CATALOG you can
    // grab (our known-good picks, verified ones badged). Anything else: Browse HF.
    var installed = catalog.filter(function (m) { return isOnDisk(m) && matchesQuery(m); });
    var knownGood = catalog.filter(function (m) { return m.curated === true && !isOnDisk(m) && matchesQuery(m); });
    // Add disk models NOT in the catalog (a plain HF repo you downloaded, e.g. Ornith) — else
    // they vanish from this page despite being on disk + launchable. Mirrors the launch dropdown.
    var inInstalled = {};
    installed.forEach(function (m) { if (m.hf_repo) inInstalled[m.hf_repo] = true; if (m.id) inInstalled[m.id] = true; });
    (self.state.downloadedList || []).forEach(function (m) {
      var repo = m.hf_repo || m.id;
      if (!repo || inInstalled[repo]) return;
      var sz = m.size_gb || m.local_size_gb || 0;
      var entry = { id: repo, slug: m.id || repo, name: m.name || repo.split('/').pop(),
        size: '~' + Math.round(sz) + ' GB', sizeGb: sz, desc: m.description || 'Downloaded model',
        quantization: m.quantization || null, minMem: m.min_memory_gb || sz, hf_repo: repo,
        downloaded: true, nodes: m.nodes || [],
        complete: m.complete !== false, incompleteReason: m.incomplete_reason || '' };
      if (matchesQuery(entry)) installed.push(entry);
    });
    var bySize = function (a, b) { return (a.sizeGb || 0) - (b.sizeGb || 0); };
    installed.sort(bySize);
    knownGood.sort(bySize);

    function cardHtml(model) {
      var isLoaded = loaded.includes(model.slug) || loaded.includes(model.id);
      var onDisk = isOnDisk(model);
      var fits = gpuMem >= (model.minMem || model.sizeGb);
      var fitBadge = self.placementBadge(model);
      // "On disk" was answering "is there a directory", which an interrupted
      // download also satisfies — so a half-downloaded model looked ready and
      // failed minutes into a launch with something about safetensors.
      var partial = onDisk && model.complete === false;
      var stops = (self.state.blockedModels || {})[model.hf_repo || model.id] || 0;
      var blockedBadge = (!isLoaded && stops) ?
        '<span class="model-badge failed" title="The memory guard stopped this ' +
        'model ' + stops + ' time' + (stops === 1 ? '' : 's') + '. Launching it ' +
        'the same way is refused — unlock it under Settings → Memory Guard.">' +
        'Blocked</span>' : '';
      // The way out of Incomplete, on the card that says Incomplete. It used
      // to live only on a paused job in the Downloads queue, which is gone
      // after a restart — so a model interrupted by a dropped link had no
      // button at all and the only offered route was deleting it.
      var resumeBtn = partial ?
        '<button class="btn-ghost model-btn-sm" data-resume-download="' +
        self.esc(model.hf_repo || model.id) + '" title="Check every file ' +
        'against the Hub, delete what is short, fetch what is missing.">' +
        'Resume download</button>' : '';
      var statusBadge = isLoaded ?
        '<span class="model-badge loaded">Loaded</span>' :
        (partial ? '<span class="model-badge failed" title="' +
            self.esc(model.incompleteReason || 'The download was interrupted') +
            '">Incomplete</span>' :
          (onDisk ? '<span class="model-badge loaded">On disk</span>' :
            '<span class="model-badge available">Available</span>'));
      var verBadge = model.verified ? '<span class="fit-badge rec">✓ Verified on GB10</span>'
        : (model.curated ? '<span class="fit-badge untested">Curated · untested</span>' : '');
      var quantBadge = model.quantization ? '<span class="fit-badge quant">' + self.esc(model.quantization.toUpperCase()) + '</span>' : '';
      // Which engine this needs. Without it the page shows a picture model
      // and a chat model as the same kind of thing, and the first hint that
      // they are not is a failed launch.
      var modalityBadge = model.modality === 'image'
        ? '<span class="fit-badge">\u25a3 Image</span>' : '';
      var paramsText = model.params ? model.params + ' params' : '';
      var descParts = [paramsText, model.size].filter(Boolean);
      var capabilityBadges = self.renderCapabilityBadges(model);
      // Where the weights actually are. "On disk" used to mean this node's
      // disk; the page now lists what the whole cluster holds, and a badge
      // that does not say which node would be the same half-truth in the
      // other direction.
      var holders = self.nodesHolding(model);
      var whereBadge = holders.length
        ? '<span class="fit-badge">' + self.esc(holders.map(function (n) {
            return self.nodeLabel(n);
          }).join(' + ')) + '</span>'
        : '';
      var actionBtn = onDisk
        ? '<button class="btn-sm downloads-delete-btn" data-model-id="' +
          self.esc(model.hf_repo || model.id) + '" data-nodes="' +
          self.esc(holders.join(',')) + '">Delete</button>'
        : '<button class="btn-sm downloads-download-btn" data-model-id="' + self.esc(model.hf_repo || model.id) + '">Download</button>';
      var shardBtn = '';
      var needsCluster = !fits || (model.proven_tp || 1) > 1;
      if (needsCluster && clusterNodeCount > 1 && totalClusterMem >= model.sizeGb && !onDisk) {
        shardBtn = '<button class="btn-sm downloads-shard-btn" data-model-id="' + self.esc(model.id) + '">Shard Across Cluster</button>';
      }
      var detailsBtn = '<button class="btn-sm download-details-btn" data-info-repo="' + self.esc(model.hf_repo || model.id) + '">Details</button>';
      return '<div class="download-card" data-model-id="' + self.esc(model.hf_repo || model.id) + '">' +
        '<div class="download-card-main">' +
        '<div class="download-card-info">' +
        '<div class="download-card-header">' +
        '<div class="download-card-name">' + self.esc(model.name || model.id) + '</div>' +
        '<div class="download-card-badges">' + verBadge + modalityBadge + quantBadge + capabilityBadges + fitBadge + statusBadge + blockedBadge + whereBadge + '</div>' +
        '</div>' +
        '<div class="download-card-repo">' + self.esc(model.hf_repo || model.id) + '</div>' +
        '<div class="download-card-desc">' + descParts.join(' &middot; ') + (model.desc ? '<br><span class="download-card-tagline">' + self.esc(model.desc) + '</span>' : '') + '</div>' +
        '</div>' +
        '<div class="download-card-actions">' + detailsBtn + resumeBtn + actionBtn + shardBtn + '</div>' +
        '</div>' +
        '</div>';
    }

    function sectionHtml(title, sub, items, emptyMsg) {
      var body = items.length
        ? '<div class="downloads-grid">' + items.map(cardHtml).join('') + '</div>'
        : '<div class="downloads-empty">' + emptyMsg + '</div>';
      return '<section class="downloads-section">' +
        '<div class="downloads-section-title">' + title +
        ' <span class="downloads-section-count">' + items.length + '</span>' +
        ' <span class="downloads-section-sub">' + sub + '</span></div>' +
        body + '</section>';
    }

    // Toolbar once; only #downloads-results re-renders on search (keeps focus).
    var needsToolbar = !container.querySelector('#downloads-search');
    if (needsToolbar) {
      container.innerHTML =
        '<div class="downloads-header downloads-actions-row">' +
        '<button id="browse-hf-btn" class="btn-sm">🤗 Browse Hugging Face</button>' +
        '<button id="import-model-btn" class="btn-sm">📥 Import from files</button>' +
        '</div>' +
        '<div id="import-panel" class="downloads-queue" style="display:none"></div>' +
        '<div id="downloads-queue" class="downloads-queue"></div>' +
        '<div class="downloads-toolbar">' +
        '<input type="text" id="downloads-search" class="search-input" placeholder="Filter your models and catalog..." value="' + this.esc(this.state.modelsSearch) + '">' +
        '</div>' +
        '<div id="downloads-results"></div>';
    }

    var catalogEmpty = query ? 'No catalog models match.'
      : 'You already have every curated model. Use 🤗 Browse Hugging Face to grab anything else.';

    var resultsContainer = container.querySelector('#downloads-results');
    if (resultsContainer) {
      resultsContainer.innerHTML =
        sectionHtml('Installed', 'on your nodes — what you have', installed,
          query ? 'No installed models match.' : 'Nothing downloaded yet. Grab one from the catalog below.') +
        sectionHtml('Catalog', 'curated known-good picks — ready to download', knownGood, catalogEmpty);
    }

    // Bind search once.
    var searchInput = document.getElementById('downloads-search');
    if (searchInput && !searchInput.dataset.bound) {
      searchInput.dataset.bound = '1';
      searchInput.addEventListener('input', function () {
        self.state.modelsSearch = searchInput.value;
        self.renderDownloads();
      });
    }

    var importBtn = document.getElementById('import-model-btn');
    if (importBtn && !importBtn.dataset.bound) {
      importBtn.dataset.bound = '1';
      importBtn.addEventListener('click', function () { self.toggleImportPanel(); });
    }

    // Bind Browse HuggingFace.
    var hfBtn = document.getElementById('browse-hf-btn');
    if (hfBtn && !hfBtn.dataset.bound) {
      hfBtn.dataset.bound = '1';
      hfBtn.addEventListener('click', function () {
        self.state.modelsFilter = 'huggingface';
        self.renderDownloads();
      });
    }

    // Bind download/delete/details + queue.
    self.bindRepoDownloadButtons(container);
    self.renderDownloadsQueue();

    // Bind shard buttons
    container.querySelectorAll('[data-resume-download]').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        self.resumeDownload(btn.getAttribute('data-resume-download'));
      });
    });
    container.querySelectorAll('.downloads-shard-btn').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var modelId = btn.dataset.modelId;
        btn.disabled = true;
        btn.textContent = 'Planning...';
        fetch('/api/sharding/plan?model=' + encodeURIComponent(modelId))
          .then(function (r) { return r.json(); })
          .then(function (data) {
            if (data.error) { self.toast(data.error, 'error'); btn.disabled = false; btn.textContent = 'Shard Across Cluster'; return; }
            var plan = data.plan;
            var sm = plan.shard_map || {};
            var h = '<div class="shard-preview">';
            h += '<div class="shard-preview-title">Sharding Plan: ' + plan.strategy + '</div>';
            h += '<div class="shard-preview-meta">World size: ' + plan.world_size + ' | TP: ' + plan.tensor_parallel_size + ' | PP: ' + plan.pipeline_parallel_size + ' | Memory: ' + plan.total_memory_required_gb + ' GB</div>';
            Object.keys(sm).forEach(function (nid) {
              var s = sm[nid];
              h += '<div class="shard-node"><span class="shard-role ' + s.role + '">' + s.role.toUpperCase() + '</span> ' +
                self.esc(nid) + '<span class="shard-detail">Layers ' + (s.layers || 'all') + ' | ~' + s.estimated_memory_gb + ' GB</span></div>';
            });
            h += '<div class="shard-actions"><button class="btn-nvidia btn-sm shard-launch-btn" data-model-id="' + self.esc(modelId) + '">Launch Sharded</button>' +
              '<button class="btn-sm shard-cancel-btn">Cancel</button></div></div>';

            var card = btn.closest('.download-card');
            var existing = card.querySelector('.shard-preview');
            if (existing) existing.remove();
            var pe = document.createElement('div');
            pe.innerHTML = h;
            card.appendChild(pe.firstChild);
            btn.disabled = false;
            btn.textContent = 'Shard Across Cluster';

            card.querySelector('.shard-launch-btn').addEventListener('click', function (ev) {
              ev.stopPropagation();
              var lb = ev.target;
              lb.disabled = true;
              lb.textContent = 'Launching...';
              fetch('/api/sharding/launch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: modelId }),
              }).then(function (r) { return r.json(); }).then(function (res) {
                if (res.error) { self.toast(res.error, 'error'); lb.disabled = false; lb.textContent = 'Launch Sharded'; }
                else { self.toast('Sharded model launching: ' + modelId, 'success'); self.refresh(); }
              }).catch(function (err) { self.toast('Error: ' + err.message, 'error'); lb.disabled = false; lb.textContent = 'Launch Sharded'; });
            });

            card.querySelector('.shard-cancel-btn').addEventListener('click', function (ev) {
              ev.stopPropagation();
              card.querySelector('.shard-preview').remove();
            });
          }).catch(function (err) { self.toast('Error: ' + err.message, 'error'); btn.disabled = false; btn.textContent = 'Shard Across Cluster'; });
      });
    });
  },
});
