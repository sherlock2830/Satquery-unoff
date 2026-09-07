import type { NextConfig } from "next";

/*
  `next build` and `next dev` both write to .next by default, so running a
  build while the dev server is up overwrites the chunks the dev server is
  serving and every request 500s with "Cannot find module './NNN.js'" until
  .next is deleted. Giving the build its own directory removes the collision
  entirely rather than relying on remembering not to do it.

  Detected from argv rather than an env var because npm scripts cannot set one
  portably on Windows without an extra dependency.
*/
// `next start` must read the directory `next build` wrote, so both use it.
const isBuild = process.argv.includes("build") || process.argv.includes("start");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  distDir: isBuild ? ".next-build" : ".next",

  // The FastAPI service in serve/ owns orchestration and inference; the site
  // proxies to it so the browser never needs a second origin (and so CORS
  // never enters the picture in dev or behind a single reverse proxy).
  async rewrites() {
    const api = process.env.SATQUERY_API ?? "http://127.0.0.1:8000";
    return [{ source: "/api/:path*", destination: `${api}/:path*` }];
  },
};

export default nextConfig;
