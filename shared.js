/* ── HoloAssist — Shared JS ── */
const _GH_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>';

function buildDock(items, opts) {
    opts = opts || {};
    var BASE = opts.base || 64, MAG = opts.mag || 88, DIST = opts.dist || 160;

    var wrap = document.createElement('aside');
    wrap.className = 'dock-wrap';
    var nav = document.createElement('nav');
    nav.className = 'dock';
    nav.setAttribute('aria-label', 'Page navigation');

    var links = items.map(function(item) {
        var a = document.createElement('a');
        a.className = 'di';
        a.href = item.href;
        if (item.id) a.dataset.id = item.id;
        a.setAttribute('aria-label', item.label);
        a.innerHTML = '<span class="di-icon">' + item.svg + '</span><span class="di-label">' + item.label + '</span>';
        nav.appendChild(a);
        return a;
    });

    nav.appendChild(Object.assign(document.createElement('div'), { className: 'd-sep' }));

    var ghLink = document.createElement('a');
    ghLink.className = 'di gh';
    ghLink.href = 'https://github.com/Sebastian-Baudille/HoloAssist-AI';
    ghLink.target = '_blank'; ghLink.rel = 'noreferrer';
    ghLink.setAttribute('aria-label', 'GitHub');
    ghLink.innerHTML = '<span class="di-icon">' + _GH_SVG + '</span><span class="di-label">GitHub</span>';
    nav.appendChild(ghLink);

    wrap.appendChild(nav);
    document.body.appendChild(wrap);

    var MAX_SCALE = MAG / BASE;

    var sectionLinks = links.filter(function(l) { return l.dataset.id; });
    if (sectionLinks.length) {
        var io = new IntersectionObserver(function(es) {
            es.forEach(function(e) {
                if (e.isIntersecting)
                    sectionLinks.forEach(function(l) { l.classList.toggle('active', l.dataset.id === e.target.id); });
            });
        }, { threshold: 0.35 });
        document.querySelectorAll('section[id]').forEach(function(s) { io.observe(s); });
    }

    function applyMag(el, inf) {
        if (inf <= 0.002) { el.style.transform = ''; return; }
        var s = 1 + (MAX_SCALE - 1) * inf * inf * (3 - 2 * inf);
        var tx = (s - 1) * 28;
        el.style.transform = 'translateX(' + tx.toFixed(1) + 'px) scale(' + s.toFixed(3) + ')';
    }
    function reset() { [].slice.call(nav.querySelectorAll('.di')).forEach(function(l) { l.style.transform = ''; }); }

    nav.addEventListener('pointermove', function(e) {
        if (window.innerWidth <= 1024) return;
        [].slice.call(nav.querySelectorAll('.di')).forEach(function(l) {
            var r = l.getBoundingClientRect(), cy = r.top + r.height / 2;
            var inf = Math.max(0, 1 - Math.abs(e.clientY - cy) / DIST);
            applyMag(l, inf);
        });
    });
    nav.addEventListener('pointerleave', reset);
    window.addEventListener('resize', reset);
    reset();
}

function initImageLightbox() {
    if (document.querySelector('.img-lightbox')) return;

    var overlay = document.createElement('div');
    overlay.className = 'img-lightbox';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', 'Expanded image');
    overlay.innerHTML =
        '<button class="img-lightbox-close" type="button" aria-label="Close image">&times;</button>' +
        '<img class="img-lightbox-img" alt="">' +
        '<div class="img-lightbox-caption"></div>';
    document.body.appendChild(overlay);

    var fullImg = overlay.querySelector('.img-lightbox-img');
    var caption = overlay.querySelector('.img-lightbox-caption');
    var closeBtn = overlay.querySelector('.img-lightbox-close');

    function close() {
        overlay.classList.remove('open');
        document.body.classList.remove('lightbox-open');
        fullImg.removeAttribute('src');
        caption.textContent = '';
    }

    function open(img) {
        var src = img.currentSrc || img.src;
        if (!src || img.closest('.img-lightbox') || img.offsetParent === null) return;
        fullImg.src = src;
        fullImg.alt = img.alt || '';
        caption.textContent = img.alt || '';
        overlay.classList.add('open');
        document.body.classList.add('lightbox-open');
        closeBtn.focus();
    }

    document.addEventListener('click', function(e) {
        var img = e.target.closest && e.target.closest('img');
        if (!img && e.target.closest) {
            var holder = e.target.closest('.photo-slot, .result-shot');
            if (holder) img = holder.querySelector('img');
        }
        if (!img) return;
        if (img.closest('.img-lightbox')) return;
        e.preventDefault();
        open(img);
    });

    closeBtn.addEventListener('click', close);
    overlay.addEventListener('click', function(e) {
        if (e.target === overlay) close();
    });
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && overlay.classList.contains('open')) close();
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initImageLightbox);
} else {
    initImageLightbox();
}
