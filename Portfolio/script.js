document.addEventListener('DOMContentLoaded', function () {

    /* ---------- Animated glyph background ---------- */
    (function glyphBackground() {
        var canvas = document.getElementById('glyph-canvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        var glyphChars = ['0', '1', '{', '}', '<', '/', '>', 'λ', 'Σ', 'π', '∞', '=', 'AI'];
        var colors = ['#8b5cf6', '#60a5fa', '#a78bfa'];
        var glyphs = [];
        var scrollY = window.scrollY || window.pageYOffset;
        var width, height;

        function sizeCanvas() {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        }

        function makeGlyphs() {
            glyphs = [];
            // Density scales gently with viewport area, capped for performance.
            var count = Math.min(70, Math.floor((width * height) / 26000));

            for (var i = 0; i < count; i++) {
                glyphs.push({
                    char: glyphChars[Math.floor(Math.random() * glyphChars.length)],
                    x: Math.random() * width,
                    y: Math.random() * height,
                    size: 11 + Math.random() * 16,
                    depth: 0.15 + Math.random() * 0.85,   // parallax + drift-speed factor
                    baseOpacity: 0.05 + Math.random() * 0.14,
                    color: colors[Math.floor(Math.random() * colors.length)],
                    drift: 0.06 + Math.random() * 0.18
                });
            }
        }

        function draw() {
            ctx.clearRect(0, 0, width, height);

            for (var i = 0; i < glyphs.length; i++) {
                var g = glyphs[i];
                var parallaxShift = (scrollY * g.depth * 0.12) % (height + 60);
                var y = g.y - parallaxShift;

                // Wrap glyphs smoothly from bottom to top as they drift/parallax past the edge
                var wrappedY = ((y % (height + 60)) + (height + 60)) % (height + 60) - 30;

                ctx.font = g.size + 'px Consolas, monospace';
                ctx.fillStyle = g.color;
                ctx.globalAlpha = g.baseOpacity;
                ctx.fillText(g.char, g.x, wrappedY);
            }

            ctx.globalAlpha = 1;
        }

        function tick() {
            for (var i = 0; i < glyphs.length; i++) {
                glyphs[i].y += glyphs[i].drift;
            }
            draw();
            rafId = requestAnimationFrame(tick);
        }

        var rafId = null;

        sizeCanvas();
        makeGlyphs();
        draw();

        if (!reduceMotion) {
            rafId = requestAnimationFrame(tick);

            window.addEventListener('scroll', function () {
                scrollY = window.scrollY || window.pageYOffset;
            }, { passive: true });
        }

        window.addEventListener('resize', function () {
            sizeCanvas();
            makeGlyphs();
            draw();
        });
    })();

    /* ---------- Mobile nav toggle ---------- */
    var toggle = document.querySelector('.nav-toggle');
    var navLinks = document.querySelector('.nav-links');

    if (toggle && navLinks) {
        toggle.addEventListener('click', function () {
            var isOpen = navLinks.classList.toggle('open');
            toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        });

        // Close the menu after picking a link (better on mobile)
        navLinks.querySelectorAll('a').forEach(function (link) {
            link.addEventListener('click', function () {
                navLinks.classList.remove('open');
                toggle.setAttribute('aria-expanded', 'false');
            });
        });
    }

    /* ---------- Active nav link on scroll ---------- */
    var sections = document.querySelectorAll('main section[id]');
    var navAnchors = document.querySelectorAll('.nav-links a');

    if (sections.length && navAnchors.length && 'IntersectionObserver' in window) {
        var sectionObserver = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    var id = entry.target.getAttribute('id');
                    navAnchors.forEach(function (a) {
                        a.classList.toggle('active', a.getAttribute('href') === '#' + id);
                    });
                }
            });
        }, { rootMargin: '-45% 0px -50% 0px' });

        sections.forEach(function (s) {
            sectionObserver.observe(s);
        });
    }

    /* ---------- Reveal-on-scroll for cards & roadmap items ---------- */
    var revealTargets = document.querySelectorAll('.reveal');

    if (revealTargets.length && 'IntersectionObserver' in window) {
        var revealObserver = new IntersectionObserver(function (entries, observer) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('in-view');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.15 });

        revealTargets.forEach(function (el) {
            revealObserver.observe(el);
        });
    } else {
        // No IntersectionObserver support: just show everything
        revealTargets.forEach(function (el) {
            el.classList.add('in-view');
        });
    }

    /* ---------- Footer year ---------- */
    var yearEl = document.querySelector('[data-year]');
    if (yearEl) {
        yearEl.textContent = new Date().getFullYear();
    }
});