// static/js/player.js
// 25.12.01 dnn dahslet
async function runPlayback() {
  const resp = await fetch('/static/sequence.json');
  const sequence = await resp.json();
  const view = document.getElementById('player'); // index-local.html의 id와 일치
  const player = dashjs.SuperPlayer().create();

  // const PROXY_ORIGIN = 'http://163.152.162.202:9989'; 
  const HOST = location.hostname;         
  // const FLASK_ORIGIN = `http://${HOST}:8081`;
  // const CDN_ORIGIN = `http://${HOST}:8080`;
  const CDN_ORIGIN = `http://163.152.162.202:8080`;
  const cdnaddress = `${CDN_ORIGIN}/dash/data/`;

  // 재생/프리로드 포인터
  let playPtr = 0;   // 현재 재생 중인 아이템 index (sequence 기준)
  let nextPreloaderPtr = 0;
  const QUEUE_LEN = 5;

  let currentPid = null;

  // class별 DNN 모델 로드 완료 여부 추적 (dash.js에서도 접근)
  const classDNNReady = {};
  window.classDNNReady = classDNNReady;

  // const SEG_LEN_SEC = 5.0;
  // const FIRST_SEG_IDX = 1;
  // let lastVideoIdx = null;
  // let lastVideoQ   = null;  
  
  // ---------- 업로드 비콘 ----------
  // function beaconStart(pid, duration) {
  //   const u = new URL(`${FLASK_ORIGIN}/uploadPlayback`);
  //   u.search = new URLSearchParams({ vid: pid, duration, watch_time: 0, last_idx: '', last_q:   ''}).toString();
  //   navigator.sendBeacon(u, '');
  // }
  // function beaconEnd(pid, duration, wt) {
  //   const u = new URL(`${FLASK_ORIGIN}/uploadPlayback`);
  //   const byTimeIdx = Math.floor((view.currentTime || 0) / SEG_LEN_SEC) + FIRST_SEG_IDX;
  //   const last_idx = (Number.isFinite(lastVideoIdx) ? lastVideoIdx : byTimeIdx);
  //   const last_q   = (Number.isFinite(lastVideoQ)   ? lastVideoQ   : -1);
    
  //   u.search = new URLSearchParams({ vid: pid, duration, watch_time: wt, last_idx: String(last_idx), last_q:  String(last_q)}).toString();
  //   navigator.sendBeacon(u, '');
  // }
  
  // ---------- 현재 아이템 이벤트 바인딩 ----------
  let unbind = () => {};
  player.attachView(view);

  if (!window.DNN_STATE) {
    window.DNN_STATE = { byVid: Object.create(null), byPlayerIdx: Object.create(null) };
  }

  function mpd(pid) {
    return `${cdnaddress}${pid}/multi_resolution.mpd?v=${(performance.now()|0)}`;
  }

  function makeDnnCtx(pid, className) {
    const ready = classDNNReady[className] === true;
    if (!ready) classDNNReady[className] = false; // 진행 중 표시
    return {
      vid: pid,
      class: className,
      reqType: ready ? 'video' : 'DNN',
      complete: ready ? 1 : 0,
      resByte: 0,
      DNN_requests: []
    };
  }

  while(nextPreloaderPtr < sequence.length && nextPreloaderPtr < QUEUE_LEN) {
    const { pid, class: className } = sequence[nextPreloaderPtr];
    if (window.DNN_STATE && window.DNN_STATE.byPlayerIdx) {
      window.DNN_STATE.byPlayerIdx[nextPreloaderPtr] = makeDnnCtx(pid, className);
      console.log('[DNN][SuperP] slot=%d vid=%s class=%s', nextPreloaderPtr, pid, className);
    }
    const ret = player.attachSource(mpd(pid));
    console.log('[INIT] attachSource idx=%d pid=%s ret=%d', nextPreloaderPtr, pid, ret);
    if (ret < 0){
      console.log('[INIT] attachSource rejected');
      break;
    }
    nextPreloaderPtr += 1;
  }


  function bindForCurrent() {
    unbind(); // 중복 방지
    const cur = sequence[playPtr];
    if (!cur) return () => {};
    const { pid, duration, watch_time } = cur;
    const target = Math.max(0, Math.min(watch_time, (duration || watch_time) - 0.5));

    let seenStart = false;
    let finished  = false;
    currentPid = pid;
    window.currentPid = pid;
    window.currentPlayerIdx = playPtr; 

    function finish(tag) {
      if (finished) return;
      finished = true;

      const seen = Math.min(target, view.currentTime || 0);
      // beaconEnd(pid, duration, seen);

      unbind();
      playPtr += 1;

      if(playPtr >= sequence.length) {
        console.log('[RUN] All videos in sequence have finished. Shutting down.');
        shutdownPlayer();
        return;
      }

      player.playNext();
      unbind = bindForCurrent(); 
      // armCanPlayToPlay('switch'); // Ensure it play

      if (nextPreloaderPtr < sequence.length) {
        const { pid, class: className } = sequence[nextPreloaderPtr];
        if (window.DNN_STATE && window.DNN_STATE.byPlayerIdx) {
          window.DNN_STATE.byPlayerIdx[nextPreloaderPtr % QUEUE_LEN] = makeDnnCtx(pid, className);
        }
        const ret = player.attachSource(mpd(pid));
        console.log('[RUN] tail attachSource idx=%d pid=%s ret=%d', nextPreloaderPtr, pid, ret);
        if (ret < 0){
          console.log('[RUN] tail attachSource rejected');
        } else {
          nextPreloaderPtr += 1;
        }
      }

    }

    function shutdownPlayer() {
      console.log('[SHUTDOWN] Player is being reset and detached.');
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
        // beaconStart(pid, duration);
      }
    }
    function onTime(e) { if (!finished && e.time >= target) finish('time'); }
    function onEnded() { finish('ended'); }
    // function onError(ev){ console.warn('[player] error', ev); finish('error'); }
    function onError(ev) {
      const err = ev && ev.error;
      console.warn('[player] ERROR event:', ev);

      if (err) {
        console.warn('[player] code=', err.code, 'msg=', err.message, 'data=', err.data);
      }
    }


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
  
    function armCanPlayToPlay(reason) {
        const onReady = () => {
            console.log('[HTML5] canplay→play (%s)', reason, 'rs=', view.readyState, 'src=', view.currentSrc);
            view.play().catch(err => console.warn('[HTML5] play() 호출 실패', err));
        };
        view.addEventListener('canplay', onReady, { once: true });
    }

  // ---------- ★★★ CORRECTED START ROUTINE ★★★ ----------
  unbind = bindForCurrent();
  player.playNext();
  armCanPlayToPlay('first');
}
document.addEventListener('DOMContentLoaded', runPlayback);