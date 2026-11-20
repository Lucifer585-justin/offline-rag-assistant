/**
 * sw.js — Service Worker
 *
 * 目的：让这个页面能"添加到主屏幕"后像 App 一样打开（PWA）。
 *
 * 缓存策略（关键，写错会导致改了代码页面不更新）：
 *   /api/*         → 不缓存，直接走网络（对话结果绝不能吃缓存）
 *   页面 HTML      → network-first：先请求，失败才用缓存（保证更新立刻生效）
 *   静态 vendor    → cache-first：Tailwind/marked/highlight 不会变，缓存最划算
 *
 * 版本号变更时旧缓存会被清掉（activate 阶段），这是 PWA 更新的标准做法。
 */

const VERSION = 'rag-assistant-v1';
const STATIC_ASSETS = [
  '/static/vendor/tailwind.js',
  '/static/vendor/marked.min.js',
  '/static/vendor/purify.min.js',
  '/static/vendor/highlight.min.js',
  '/static/vendor/github-dark.min.css',
  '/static/icon.svg',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(VERSION)
      .then(cache => cache.addAll(STATIC_ASSETS).catch(() => null))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== VERSION).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // 只处理同源 GET
  if (event.request.method !== 'GET' || url.origin !== self.location.origin) return;

  // 接口一律不缓存
  if (url.pathname.startsWith('/api/')) return;

  // 页面：network-first，断网时回退缓存
  if (event.request.mode === 'navigate' || url.pathname === '/') {
    event.respondWith(
      fetch(event.request)
        .then(resp => {
          const copy = resp.clone();
          caches.open(VERSION).then(c => c.put(event.request, copy)).catch(() => {});
          return resp;
        })
        .catch(() => caches.match(event.request).then(r => r || caches.match('/')))
    );
    return;
  }

  // 静态资源：cache-first
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(event.request).then(cached => cached || fetch(event.request).then(resp => {
        const copy = resp.clone();
        caches.open(VERSION).then(c => c.put(event.request, copy)).catch(() => {});
        return resp;
      }))
    );
  }
});
