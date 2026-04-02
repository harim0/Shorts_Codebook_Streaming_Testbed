// static/js/player.js
// 25.11.29 preload...attachSource...... 수정 (슬롯 제대로 바꾸도록)

// static/js/player.js
async function runPlayback() {
  const resp = await fetch('/static/sequence.json');
  const sequence = await resp.json();
  const view = document.getElementById('player'); // index-local.html의 id와 일치
  const player = dashjs.SuperPlayer().create();

  // const PROXY_ORIGIN = 'http://163.152.162.202:9989'; 
  const HOST = location.hostname;         
  const PROXY_ORIGIN = `http://${HOST}:9989`;
  const cdnaddress = `${PROXY_ORIGIN}/dash/data/`;

  // 재생/프리로드 포인터
  let playPtr = 0;   // 현재 재생 중인 아이템 index (sequence 기준)
  let nextPreloaderPtr = 0;
  const QUEUE_LEN = 5;

  const SEG_LEN_SEC = 5.0;
  const FIRST_SEG_IDX = 1;
  let currentPid = null;
  let lastVideoIdx = null;
  let lastVideoQ   = null;  
  
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
    const last_q   = (Number.isFinite(lastVideoQ)   ? lastVideoQ   : -1);
    
    u.search = new URLSearchParams({ vid: pid, duration, watch_time: wt, last_idx: String(last_idx), last_q:  String(last_q)}).toString();
    navigator.sendBeacon(u, '');
  }
  
  // ---------- 현재 아이템 이벤트 바인딩 ----------
  let unbind = () => {};
  player.attachView(view);
  
  function mpd(pid) {
    return `${cdnaddress}${pid}/manifest.mpd?v=${(performance.now()|0)}`;
  }

  while(nextPreloaderPtr < sequence.length && nextPreloaderPtr < QUEUE_LEN) {
    const { pid } = sequence[nextPreloaderPtr];
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

    function finish(tag) {
      if (finished) return;
      finished = true;

      const seen = Math.min(target, view.currentTime || 0);
      beaconEnd(pid, duration, seen);

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
        const { pid } = sequence[nextPreloaderPtr];
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