// static/js/player.js
// dnn single

async function runPlayback() {
  const resp = await fetch('/static/sequence.json');
  const sequence = await resp.json();
  const view = document.getElementById('player'); // index-local.html의 id와 일치
  const player = dashjs.SuperPlayer().create();

  window.DNN_selection - 3;
  window.SRProcess_enabled - true;
  window.DNNTest_enabled - false;
  window.DNN_donload_enabled - false;


  // const PROXY_ORIGIN = 'http://163.152.162.202:9989'; 
  const HOST = location.hostname;         
  const PROXY_ORIGIN = `http://${HOST}:9989`;
  const cdnaddress = `${PROXY_ORIGIN}/dash/data/`;

  // 재생/프리로드 포인터
  let playPtr = 0;   // 현재 재생 중인 아이템 index (sequence 기준)
  const SEG_LEN_SEC = 5.0;
  const FIRST_SEG_IDX = 1;
  let currentPid = null;
  let lastVideoIdx = null;
  let lastVideoQ   = null;  

  function beaconRebuffer(pid, dur) {
    const u = new URL(`${PROXY_ORIGIN}/uploadRebuffer`);
    u.search = new URLSearchParams({ vid: pid, d: dur }).toString();
    navigator.sendBeacon(u, '');
  }

  player.attachView(view);

  view.crossOrigin = 'anonymous';
  view.muted = true;      
  view.playsInline = true;
  view.setAttribute('playsinline','');
  
  player.on(dashjs.MediaPlayer.events.STREAM_INITIALIZED, () => console.log('[dash] STREAM_INITIALIZED'));
  player.on(dashjs.MediaPlayer.events.PLAYBACK_STARTED,   () => console.log('[dash] PLAYBACK_STARTED'));
  
  // streamProcessor 생성 여부
  const EB = dashjs.__internals?.eventBus || player;
  EB.on('streamInitializing', e => {
    console.log('[EB] STREAM_INITIALIZING', e.mediaType, e.streamInfo);
  });
  EB.on('streamInitialized', e => {
    console.log('[EB] STREAM_INITIALIZED', e.streamInfo);
  });

  // manifest 파싱 직후 AdaptationSet 확인
  player.on(dashjs.MediaPlayer.events.MANIFEST_LOADED, e => {
      console.log('[MPD] manifest loaded');
      console.log('video tracks:', player.getTracksFor('video'));
      console.log('audio tracks:', player.getTracksFor('audio'));
  });

  // 오디오 요청 발생 시점 (ScheduleController → FragmentLoader)
  EB.on('mediaFragmentNeeded', e => {
      console.log('[EB] MEDIA_FRAGMENT_NEEDED', e.mediaType, e.requestedTime);
  });
  EB.on('mediaFragmentLoadingStarted', e => {
      console.log('[EB] MEDIA_FRAGMENT_LOADING_STARTED', e.request.mediaType, e.request.url);
  });
  EB.on('mediaFragmentLoaded', e => {
      console.log('[EB] MEDIA_FRAGMENT_LOADED', e.request.mediaType, e.request.url);
  });

  player.on(dashjs.MediaPlayer.events.QUALITY_CHANGE_REQUESTED, e => {
  console.log('[QCR]', e.mediaType, 'from', e.oldQuality, 'to', e.newQuality, e.reason);
  });
  player.on(dashjs.MediaPlayer.events.QUALITY_CHANGE_RENDERED, e => {
    console.log('[QCRD]', e.mediaType, 'to', e.newQuality);
  });

  const E = dashjs.MediaPlayer.events;
  const seen = { audio:0, video:0 };

  player.on(E.FRAGMENT_LOADING_STARTED,  e => {
    if (!e.request || e.request.mediaType !== 'video') return;
    try {
      const u = new URL(e.request.url, location.href);
      if (currentPid) u.searchParams.set('vid', currentPid);
      if (Number.isFinite(e.request.quality)) u.searchParams.set('q', String(e.request.quality));
      if (Number.isFinite(e.request.index))   u.searchParams.set('idx', String(e.request.index));
      e.request.url = u.toString(); 
    } catch (_) {}
  });
  player.on(E.FRAGMENT_LOADING_COMPLETED, e => {
    const t = e.request?.mediaType;
    const bytes = e.response ? (e.response.byteLength||e.response.length||'') : '';
    console.log('[EV] DONE', t, e.request?.url, 'bytes=', bytes, 'err=', e.error||null);
    if (e.request?.mediaType === 'video') {
      console.log('[SEG]', 'idx=', e.request.index, 'dur=', e.request.duration, 'bytes=', bytes, '[t=',t,']');
    }
    if (t) seen[t]++; 
    if (!e.request || e.request.mediaType !== 'video') return;
    if (Number.isFinite(e.request.index))   lastVideoIdx = e.request.index;
    if (Number.isFinite(e.request.quality)) lastVideoQ   = e.request.quality;
  });

  if (!view.__html5Bound) {
    const log = (name, ...rest) => console.log(`[HTML5] ${name}`, ...rest);
    view.addEventListener('pause',   () => log('pause'));
    let rbStart = null;
    let vidAtWait = null;

    view.addEventListener('waiting', () => {
      // HTML5 video가 재생을 못 하고 '버퍼링 대기' 상태로 들어간 시각(초)
      rbStart = performance.now()/1000;
      // 끊김을 유발한 영상 id 고정(재생 아이템이 바뀌어도 안전)
      vidAtWait = sequence[playPtr]?.pid;
    });
    view.addEventListener('playing', () => {
      // '대기→재생 재개'가 되는 순간, 방금 끊긴 구간의 길이를 계산
      if (rbStart != null) {
        const dur = performance.now()/1000 - rbStart;  // 끊긴 시간(초)
        // 서버에 "이번 끊김은 dur초였다"라는 한 줄을 비콘으로 보냄(비동기·non-blocking)
        if (dur > 0.05) {
          beaconRebuffer(vidAtWait || sequence[playPtr].pid, dur);
        }
        rbStart = null;
        vidAtWait = null;
      }
    });
    view.addEventListener('error',   () => log('error', view.error));
    view.addEventListener('loadedmetadata', () => console.log('[HTML5] loadedmetadata', view.readyState));
    view.addEventListener('canplay',        () => console.log('[HTML5] canplay', view.readyState));
    view.addEventListener('canplaythrough', () => console.log('[HTML5] canplaythrough', view.readyState));
    view.addEventListener('timeupdate', () => log('t', view.currentTime.toFixed(2)));
    view.__html5Bound = true; 
  }
  view.play().then(()=>console.log('play OK')).catch(err => console.warn('play() blocked', err && err.name, err));

  setInterval(() => {
    const b = view.buffered;
    const start = b.length ? b.start(0).toFixed(2) : '∅';
    const end   = b.length ? b.end(b.length - 1).toFixed(2) : '∅';
    console.log('[HTML5]',
      'src=', view.currentSrc || '(empty)',
      'net=', view.networkState, 'rs=', view.readyState,
      'buffer=', `${start}~${end}`,
      't=', (view.currentTime||0).toFixed(2),
      'WH=', view.videoWidth, 'x', view.videoHeight);
  }, 10000);

  function mpd(pid) {
    return `${cdnaddress}${pid}/multi_resolution.mpd?v=${(performance.now()|0)}`;
  }

  // ---------- 업로드 비콘 ----------
  function beaconStart(pid, duration) {
    const u = new URL(`${PROXY_ORIGIN}/uploadPlayback`);
    u.search = new URLSearchParams({ vid: pid, duration, watch_time: 0, last_idx: '', last_q:   ''}).toString();
    navigator.sendBeacon(u, '');
  }
  function beaconEnd(pid, duration, wt) {
    const u = new URL(`${PROXY_ORIGIN}/uploadPlayback`);
    const byTimeIdx = Math.floor((view.currentTime || 0) / SEG_LEN_SEC) + FIRST_SEG_IDX;
    const last_idx = (Number.isFinite(lastVideoIdx) ? lastVideoIdx : byTimeIdx);
    const last_q   = (Number.isFinite(lastVideoQ)   ? lastVideoQ   : player.getQualityFor('video'));

    u.search = new URLSearchParams({ vid: pid, duration, watch_time: wt, last_idx: String(last_idx), last_q:  String(last_q)}).toString();
    navigator.sendBeacon(u, '');
  }

  // ---------- 현재 아이템 이벤트 바인딩 ----------
  let unbind = () => {};

  function bindForCurrent() {
    unbind(); // 중복 방지
    const cur = sequence[playPtr];
    if (!cur) return () => {};
    const { pid, duration, watch_time } = cur;
    const target = Math.max(0, Math.min(watch_time, (duration || watch_time) - 0.5));

    let seenStart = false;
    let finished  = false;
    currentPid = pid;

    function finish(tag) {
      if (finished) return;
      finished = true;
      const seen = Math.min(target, view.currentTime || 0);
      beaconEnd(pid, duration, seen);
      console.log('[RUN] finish(%s) pid=%s -> next playPtr=%d', tag, pid, playPtr + 1);
      
      unbind();

      playPtr += 1;
      // if (playPtr < 2) { // 품질 로직 작동 확인용 2개만
      if (playPtr < sequence.length) {
        // 다음 슬롯으로 view 붙이고, 다음 소스 직결 + canplay에서 play
        player.playNext();
        player.attachSourceForCurrent(mpd(sequence[playPtr].pid));
        armCanPlayToPlay('switch'); // Ensure it plays
        unbind = bindForCurrent(); 
      } else {
        console.log('[RUN] All videos in sequence have finished. Shutting down.');
        shutdownPlayer();
      }
    }

    function shutdownPlayer() {
      console.log('[SHUTDOWN] Player is being reset and detached.');
      try {
        clearInterval(prefetchTick);
      } catch {}
      try {
        player.reset(); // dash.js의 모든 리소스를 정리합니다.
      } catch {}
      try {
        const el = document.getElementById('player');
        if (el) {
          el.removeAttribute('src'); // blob: URL 해제
          el.load(); // MSE 해제 유도
        }
      } catch (e) {
        console.warn('[SHUTDOWN] Player teardown failed:', e);
      }
    }

    function onPlaying() {
      if (!seenStart) {
        seenStart = true;
        beaconStart(pid, duration);
      }
    }
    function onTime(e) { if (!finished && e.time >= target) finish('time'); }
    function onEnded() { finish('ended'); }
    function onError(ev){ console.warn('[player] error', ev); finish('error'); }

    player.on(dashjs.MediaPlayer.events.PLAYBACK_PLAYING, onPlaying);
    player.on(dashjs.MediaPlayer.events.PLAYBACK_TIME_UPDATED, onTime);
    player.on(dashjs.MediaPlayer.events.PLAYBACK_ENDED, onEnded);
    player.on(dashjs.MediaPlayer.events.ERROR, onError);

    const newUnbind = () => {
      player.off(dashjs.MediaPlayer.events.PLAYBACK_PLAYING, onPlaying);
      player.off(dashjs.MediaPlayer.events.PLAYBACK_TIME_UPDATED, onTime);
      player.off(dashjs.MediaPlayer.events.PLAYBACK_ENDED, onEnded);
      player.off(dashjs.MediaPlayer.events.ERROR, onError);
    };
    return newUnbind;
  }
  // ★ Utility to reliably trigger play() on canplay event (avoids race conditions)
    function armCanPlayToPlay(reason) {
        const onReady = () => {
            console.log('[HTML5] canplay→play (%s)', reason, 'rs=', view.readyState, 'src=', view.currentSrc);
            view.play().catch(err => console.warn('[HTML5] play() 호출 실패', err));
        };
        view.addEventListener('canplay', onReady, { once: true });
    }

  // ---------- ★★★ CORRECTED START ROUTINE ★★★ ----------
  // 1) Bind events for the very first item (playPtr = 0)
  unbind = bindForCurrent();

  // 2) Attach the view to slot 0, then attach the source for video 0.
  player.playNext();
  player.attachSourceForCurrent(mpd(sequence[playPtr].pid));

  // 3) When the video element is ready ('canplay'), then call play().
  //    This is the most reliable way to handle autoplay.
  armCanPlayToPlay('first');
}
document.addEventListener('DOMContentLoaded', runPlayback);