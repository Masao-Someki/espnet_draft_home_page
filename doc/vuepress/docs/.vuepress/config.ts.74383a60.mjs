// docs/.vuepress/config.ts
import { createRequire as createRequire2 } from "node:module";
import process from "node:process";
import { viteBundler } from "@vuepress/bundler-vite";
import { webpackBundler } from "@vuepress/bundler-webpack";
import pkg from "@vuepress/plugin-search";
import { registerComponentsPlugin } from "@vuepress/plugin-register-components";
import { shikiPlugin } from "@vuepress/plugin-shiki";
import { defaultTheme } from "@vuepress/theme-default";
import { defineUserConfig } from "vuepress";
import { getDirname, path } from "vuepress/utils";

// docs/.vuepress/configs/head.ts
var head = [
  // [
  //   'link',
  //   {
  //     rel: 'icon',
  //     type: 'image/png',
  //     sizes: '16x16',
  //     href: `/images/icons/favicon-16x16.png`,
  //   },
  // ],
  // [
  //   'link',
  //   {
  //     rel: 'icon',
  //     type: 'image/png',
  //     sizes: '32x32',
  //     href: `/images/icons/favicon-32x32.png`,
  //   },
  // ],
  ["link", { rel: "manifest", href: "/manifest.webmanifest" }],
  ["meta", { name: "application-name", content: "Example" }],
  ["meta", { name: "apple-mobile-web-app-title", content: "Example" }],
  ["meta", { name: "apple-mobile-web-app-status-bar-style", content: "black" }],
  // [
  //   'link',
  //   { rel: 'apple-touch-icon', href: `/images/icons/apple-touch-icon.png` },
  // ],
  // [
  //   'link',
  //   {
  //     rel: 'mask-icon',
  //     href: '/images/icons/safari-pinned-tab.svg',
  //     color: '#3eaf7c',
  //   },
  // ],
  ["meta", { name: "msapplication-TileColor", content: "#3eaf7c" }],
  ["meta", { name: "theme-color", content: "#3eaf7c" }]
];

// docs/.vuepress/configs/meta.ts
import { createRequire } from "node:module";
import { fs } from "vuepress/utils";
var __vite_injected_original_import_meta_url = "file:///hdd/doc/notebook/docs/.vuepress/configs/meta.ts";
var require2 = createRequire(__vite_injected_original_import_meta_url);
var version = fs.readJsonSync(
  require2.resolve("vuepress/package.json")
).version;

// docs/.vuepress/configs/navbar/en.ts
var navbarEn = [
  {
    text: "Guide",
    children: [
      "/guide/introduction.md",
      "/guide/getting-started.md",
      "/guide/configuration.md",
      "/guide/page.md",
      "/guide/markdown.md",
      "/guide/assets.md",
      "/guide/i18n.md",
      "/guide/deployment.md",
      "/guide/theme.md",
      "/guide/plugin.md",
      "/guide/bundler.md",
      "/guide/migration.md",
      "/guide/troubleshooting.md"
    ]
  },
  {
    text: "Reference",
    children: [
      {
        text: "VuePress",
        children: [
          {
            text: "CLI",
            link: "/reference/cli.html"
          },
          "/reference/config.md",
          "/reference/frontmatter.md",
          "/reference/components.md",
          "/reference/plugin-api.md",
          "/reference/theme-api.md",
          "/reference/client-api.md",
          "/reference/node-api.md"
        ]
      },
      {
        text: "Bundlers",
        children: [
          "/reference/bundler/vite.md",
          "/reference/bundler/webpack.md"
        ]
      }
    ]
  },
  {
    text: "Learn More",
    children: [
      {
        text: "Advanced",
        children: [
          "/advanced/architecture.md",
          "/advanced/plugin.md",
          "/advanced/theme.md",
          {
            text: "Cookbook",
            link: "/advanced/cookbook/"
          }
        ]
      },
      {
        text: "Resources",
        children: [
          {
            text: "Official Ecosystem",
            link: "https://ecosystem.vuejs.press/"
          },
          {
            text: "VuePress MarketPlace",
            link: "https://marketplace.vuejs.press"
          },
          {
            text: "Contributing Guide",
            link: "https://github.com/vuepress/core/blob/main/CONTRIBUTING.md"
          }
        ]
      }
    ]
  },
  {
    text: `v${version}`,
    children: [
      {
        text: "Changelog",
        link: "https://github.com/vuepress/core/blob/main/CHANGELOG.md"
      },
      {
        text: "v1.x",
        link: "https://v1.vuepress.vuejs.org"
      },
      {
        text: "v0.x",
        link: "https://v0.vuepress.vuejs.org"
      }
    ]
  }
];

// docs/.vuepress/config.ts
import { load } from "js-yaml";
import { readFileSync } from "fs";
var __vite_injected_original_import_meta_url2 = "file:///hdd/doc/notebook/docs/.vuepress/config.ts";
var { searchPlugin } = pkg;
var __dirname = getDirname(__vite_injected_original_import_meta_url2);
var require3 = createRequire2(__vite_injected_original_import_meta_url2);
var isProd = process.env.NODE_ENV === "production";
var navbarEn2 = load(readFileSync("menubars.yml", "utf-8"));
var sidebarEn = load(readFileSync("sidebars.yml", "utf-8"));
var config_default = defineUserConfig({
  // set site base to default value
  base: "/",
  // extra tags in `<head>`
  head,
  // site-level locales config
  locales: {
    "/": {
      lang: "en-US",
      title: "ESPnet-Example",
      description: "Example notebooks for ESPnet"
    }
  },
  // specify bundler via environment variable
  bundler: process.env.DOCS_BUNDLER === "webpack" ? webpackBundler() : viteBundler(),
  // configure default theme
  theme: defaultTheme({
    // hostname: 'https://v2.vuepress.vuejs.org',
    logo: "/images/espnet_logo1.png",
    // repo: 'vuepress/docs',
    docsDir: "docs",
    // theme-level locales config
    locales: {
      /**
       * English locale config
       *
       * As the default locale of @vuepress/theme-default is English,
       * we don't need to set all of the locale fields
       */
      "/": {
        // navbar
        navbar: navbarEn2,
        // sidebar
        sidebar: sidebarEn,
        // page meta
        editLinkText: "Edit this page on GitHub"
      }
    },
    themePlugins: {
      // only enable git plugin in production mode
      git: isProd,
      // use shiki plugin in production mode instead
      prismjs: !isProd
    }
  }),
  // configure markdown
  markdown: {
    importCode: {
      handleImportPath: (importPath) => {
        if (importPath.startsWith("@vuepress/")) {
          const packageName = importPath.match(/^(@vuepress\/[^/]*)/)[1];
          return importPath.replace(
            packageName,
            path.dirname(require3.resolve(`${packageName}/package.json`))
          ).replace("/src/", "/lib/").replace(/hotKey\.ts$/, "hotKey.d.ts");
        }
        return importPath;
      }
    }
  },
  themeConfig: {
    searchPlaceholder: "Search..."
  },
  // use plugins
  plugins: [
    // docsearchPlugin({
    //   appId: '',
    //   apiKey: '',
    //   indexName: 'vuepress',
    //   searchParameters: {
    //     facetFilters: ['tags:v2'],
    //   },
    // }),
    searchPlugin({
      // getExtraFields: (page) => page.frontmatter.tags,
      maxSuggestions: 15,
      hotKeys: ["s", "/"],
      locales: {
        "/": {
          placeholder: "Search"
        }
      }
    }),
    registerComponentsPlugin({
      componentsDir: path.resolve(__dirname, "./components")
    }),
    // only enable shiki plugin in production mode
    isProd ? shikiPlugin({
      langs: ["bash", "diff", "json", "md", "ts", "vue"],
      theme: "dark-plus"
    }) : []
  ]
});
export {
  config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiZG9jcy8udnVlcHJlc3MvY29uZmlnLnRzIiwgImRvY3MvLnZ1ZXByZXNzL2NvbmZpZ3MvaGVhZC50cyIsICJkb2NzLy52dWVwcmVzcy9jb25maWdzL21ldGEudHMiLCAiZG9jcy8udnVlcHJlc3MvY29uZmlncy9uYXZiYXIvZW4udHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCIvaGRkL2RvYy9ub3RlYm9vay9kb2NzLy52dWVwcmVzc1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiL2hkZC9kb2Mvbm90ZWJvb2svZG9jcy8udnVlcHJlc3MvY29uZmlnLnRzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9oZGQvZG9jL25vdGVib29rL2RvY3MvLnZ1ZXByZXNzL2NvbmZpZy50c1wiO2ltcG9ydCB7IGNyZWF0ZVJlcXVpcmUgfSBmcm9tICdub2RlOm1vZHVsZSdcbmltcG9ydCBwcm9jZXNzIGZyb20gJ25vZGU6cHJvY2VzcydcbmltcG9ydCB7IHZpdGVCdW5kbGVyIH0gZnJvbSAnQHZ1ZXByZXNzL2J1bmRsZXItdml0ZSdcbmltcG9ydCB7IHdlYnBhY2tCdW5kbGVyIH0gZnJvbSAnQHZ1ZXByZXNzL2J1bmRsZXItd2VicGFjaydcbi8vIGltcG9ydCB7IGRvY3NlYXJjaFBsdWdpbiB9IGZyb20gJ0B2dWVwcmVzcy9wbHVnaW4tZG9jc2VhcmNoJ1xuaW1wb3J0IHBrZyBmcm9tICdAdnVlcHJlc3MvcGx1Z2luLXNlYXJjaCc7XG5jb25zdCB7IHNlYXJjaFBsdWdpbiB9ID0gcGtnO1xuaW1wb3J0IHsgcmVnaXN0ZXJDb21wb25lbnRzUGx1Z2luIH0gZnJvbSAnQHZ1ZXByZXNzL3BsdWdpbi1yZWdpc3Rlci1jb21wb25lbnRzJ1xuaW1wb3J0IHsgc2hpa2lQbHVnaW4gfSBmcm9tICdAdnVlcHJlc3MvcGx1Z2luLXNoaWtpJ1xuaW1wb3J0IHsgZGVmYXVsdFRoZW1lIH0gZnJvbSAnQHZ1ZXByZXNzL3RoZW1lLWRlZmF1bHQnXG5pbXBvcnQgeyBkZWZpbmVVc2VyQ29uZmlnIH0gZnJvbSAndnVlcHJlc3MnXG5pbXBvcnQgeyBnZXREaXJuYW1lLCBwYXRoIH0gZnJvbSAndnVlcHJlc3MvdXRpbHMnXG5cbmltcG9ydCB7XG4gIGhlYWQsXG4gIC8vIG5hdmJhckVuLFxuICAvLyBzaWRlYmFyRW4sXG59IGZyb20gJy4vY29uZmlncy9pbmRleC5qcydcbmltcG9ydCB7IGxvYWQgfSBmcm9tICdqcy15YW1sJ1xuaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnXG5cbmNvbnN0IF9fZGlybmFtZSA9IGdldERpcm5hbWUoaW1wb3J0Lm1ldGEudXJsKVxuY29uc3QgcmVxdWlyZSA9IGNyZWF0ZVJlcXVpcmUoaW1wb3J0Lm1ldGEudXJsKVxuY29uc3QgaXNQcm9kID0gcHJvY2Vzcy5lbnYuTk9ERV9FTlYgPT09ICdwcm9kdWN0aW9uJ1xuXG5jb25zdCBuYXZiYXJFbiA9IGxvYWQocmVhZEZpbGVTeW5jKCdtZW51YmFycy55bWwnLCAndXRmLTgnKSlcbmNvbnN0IHNpZGViYXJFbiA9IGxvYWQocmVhZEZpbGVTeW5jKCdzaWRlYmFycy55bWwnLCAndXRmLTgnKSlcblxuZXhwb3J0IGRlZmF1bHQgZGVmaW5lVXNlckNvbmZpZyh7XG4gIC8vIHNldCBzaXRlIGJhc2UgdG8gZGVmYXVsdCB2YWx1ZVxuICBiYXNlOiAnLycsXG5cbiAgLy8gZXh0cmEgdGFncyBpbiBgPGhlYWQ+YFxuICBoZWFkLFxuXG4gIC8vIHNpdGUtbGV2ZWwgbG9jYWxlcyBjb25maWdcbiAgbG9jYWxlczoge1xuICAgICcvJzoge1xuICAgICAgbGFuZzogJ2VuLVVTJyxcbiAgICAgIHRpdGxlOiAnRVNQbmV0LUV4YW1wbGUnLFxuICAgICAgZGVzY3JpcHRpb246ICdFeGFtcGxlIG5vdGVib29rcyBmb3IgRVNQbmV0JyxcbiAgICB9LFxuICB9LFxuXG4gIC8vIHNwZWNpZnkgYnVuZGxlciB2aWEgZW52aXJvbm1lbnQgdmFyaWFibGVcbiAgYnVuZGxlcjpcbiAgICBwcm9jZXNzLmVudi5ET0NTX0JVTkRMRVIgPT09ICd3ZWJwYWNrJyA/IHdlYnBhY2tCdW5kbGVyKCkgOiB2aXRlQnVuZGxlcigpLFxuXG4gIC8vIGNvbmZpZ3VyZSBkZWZhdWx0IHRoZW1lXG4gIHRoZW1lOiBkZWZhdWx0VGhlbWUoe1xuICAgIC8vIGhvc3RuYW1lOiAnaHR0cHM6Ly92Mi52dWVwcmVzcy52dWVqcy5vcmcnLFxuICAgIGxvZ286ICcvaW1hZ2VzL2VzcG5ldF9sb2dvMS5wbmcnLFxuICAgIC8vIHJlcG86ICd2dWVwcmVzcy9kb2NzJyxcbiAgICBkb2NzRGlyOiAnZG9jcycsXG5cbiAgICAvLyB0aGVtZS1sZXZlbCBsb2NhbGVzIGNvbmZpZ1xuICAgIGxvY2FsZXM6IHtcbiAgICAgIC8qKlxuICAgICAgICogRW5nbGlzaCBsb2NhbGUgY29uZmlnXG4gICAgICAgKlxuICAgICAgICogQXMgdGhlIGRlZmF1bHQgbG9jYWxlIG9mIEB2dWVwcmVzcy90aGVtZS1kZWZhdWx0IGlzIEVuZ2xpc2gsXG4gICAgICAgKiB3ZSBkb24ndCBuZWVkIHRvIHNldCBhbGwgb2YgdGhlIGxvY2FsZSBmaWVsZHNcbiAgICAgICAqL1xuICAgICAgJy8nOiB7XG4gICAgICAgIC8vIG5hdmJhclxuICAgICAgICBuYXZiYXI6IG5hdmJhckVuLFxuICAgICAgICAvLyBzaWRlYmFyXG4gICAgICAgIHNpZGViYXI6IHNpZGViYXJFbixcbiAgICAgICAgLy8gcGFnZSBtZXRhXG4gICAgICAgIGVkaXRMaW5rVGV4dDogJ0VkaXQgdGhpcyBwYWdlIG9uIEdpdEh1YicsXG4gICAgICB9LFxuICAgIH0sXG5cbiAgICB0aGVtZVBsdWdpbnM6IHtcbiAgICAgIC8vIG9ubHkgZW5hYmxlIGdpdCBwbHVnaW4gaW4gcHJvZHVjdGlvbiBtb2RlXG4gICAgICBnaXQ6IGlzUHJvZCxcbiAgICAgIC8vIHVzZSBzaGlraSBwbHVnaW4gaW4gcHJvZHVjdGlvbiBtb2RlIGluc3RlYWRcbiAgICAgIHByaXNtanM6ICFpc1Byb2QsXG4gICAgfSxcbiAgfSksXG5cbiAgLy8gY29uZmlndXJlIG1hcmtkb3duXG4gIG1hcmtkb3duOiB7XG4gICAgaW1wb3J0Q29kZToge1xuICAgICAgaGFuZGxlSW1wb3J0UGF0aDogKGltcG9ydFBhdGgpID0+IHtcbiAgICAgICAgLy8gaGFuZGxlIEB2dWVwcmVzcyBwYWNrYWdlcyBpbXBvcnQgcGF0aFxuICAgICAgICBpZiAoaW1wb3J0UGF0aC5zdGFydHNXaXRoKCdAdnVlcHJlc3MvJykpIHtcbiAgICAgICAgICBjb25zdCBwYWNrYWdlTmFtZSA9IGltcG9ydFBhdGgubWF0Y2goL14oQHZ1ZXByZXNzXFwvW14vXSopLykhWzFdXG4gICAgICAgICAgcmV0dXJuIGltcG9ydFBhdGhcbiAgICAgICAgICAgIC5yZXBsYWNlKFxuICAgICAgICAgICAgICBwYWNrYWdlTmFtZSxcbiAgICAgICAgICAgICAgcGF0aC5kaXJuYW1lKHJlcXVpcmUucmVzb2x2ZShgJHtwYWNrYWdlTmFtZX0vcGFja2FnZS5qc29uYCkpLFxuICAgICAgICAgICAgKVxuICAgICAgICAgICAgLnJlcGxhY2UoJy9zcmMvJywgJy9saWIvJylcbiAgICAgICAgICAgIC5yZXBsYWNlKC9ob3RLZXlcXC50cyQvLCAnaG90S2V5LmQudHMnKVxuICAgICAgICB9XG4gICAgICAgIHJldHVybiBpbXBvcnRQYXRoXG4gICAgICB9LFxuICAgIH0sXG4gIH0sXG4gIHRoZW1lQ29uZmlnOiB7XG4gICAgc2VhcmNoUGxhY2Vob2xkZXI6ICdTZWFyY2guLi4nXG4gIH0sXG4gIC8vIHVzZSBwbHVnaW5zXG4gIHBsdWdpbnM6IFtcbiAgICAvLyBkb2NzZWFyY2hQbHVnaW4oe1xuICAgIC8vICAgYXBwSWQ6ICcnLFxuICAgIC8vICAgYXBpS2V5OiAnJyxcbiAgICAvLyAgIGluZGV4TmFtZTogJ3Z1ZXByZXNzJyxcbiAgICAvLyAgIHNlYXJjaFBhcmFtZXRlcnM6IHtcbiAgICAvLyAgICAgZmFjZXRGaWx0ZXJzOiBbJ3RhZ3M6djInXSxcbiAgICAvLyAgIH0sXG4gICAgLy8gfSksXG4gICAgc2VhcmNoUGx1Z2luKHtcbiAgICAgIC8vIGdldEV4dHJhRmllbGRzOiAocGFnZSkgPT4gcGFnZS5mcm9udG1hdHRlci50YWdzLFxuICAgICAgbWF4U3VnZ2VzdGlvbnM6IDE1LFxuICAgICAgaG90S2V5czogWydzJywgJy8nXSxcbiAgICAgIGxvY2FsZXM6IHtcbiAgICAgICAgJy8nOiB7XG4gICAgICAgICAgcGxhY2Vob2xkZXI6ICdTZWFyY2gnLFxuICAgICAgICB9XG4gICAgICB9XG4gICAgfSksXG4gICAgcmVnaXN0ZXJDb21wb25lbnRzUGx1Z2luKHtcbiAgICAgIGNvbXBvbmVudHNEaXI6IHBhdGgucmVzb2x2ZShfX2Rpcm5hbWUsICcuL2NvbXBvbmVudHMnKSxcbiAgICB9KSxcbiAgICAvLyBvbmx5IGVuYWJsZSBzaGlraSBwbHVnaW4gaW4gcHJvZHVjdGlvbiBtb2RlXG4gICAgaXNQcm9kXG4gICAgICA/IHNoaWtpUGx1Z2luKHtcbiAgICAgICAgICBsYW5nczogWydiYXNoJywgJ2RpZmYnLCAnanNvbicsICdtZCcsICd0cycsICd2dWUnXSxcbiAgICAgICAgICB0aGVtZTogJ2RhcmstcGx1cycsXG4gICAgICAgIH0pXG4gICAgICA6IFtdLFxuICBdLFxufSlcbiIsICJjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfZGlybmFtZSA9IFwiL2hkZC9kb2Mvbm90ZWJvb2svZG9jcy8udnVlcHJlc3MvY29uZmlnc1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiL2hkZC9kb2Mvbm90ZWJvb2svZG9jcy8udnVlcHJlc3MvY29uZmlncy9oZWFkLnRzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9oZGQvZG9jL25vdGVib29rL2RvY3MvLnZ1ZXByZXNzL2NvbmZpZ3MvaGVhZC50c1wiO2ltcG9ydCB0eXBlIHsgSGVhZENvbmZpZyB9IGZyb20gJ3Z1ZXByZXNzL2NvcmUnXG5cbmV4cG9ydCBjb25zdCBoZWFkOiBIZWFkQ29uZmlnW10gPSBbXG4gIC8vIFtcbiAgLy8gICAnbGluaycsXG4gIC8vICAge1xuICAvLyAgICAgcmVsOiAnaWNvbicsXG4gIC8vICAgICB0eXBlOiAnaW1hZ2UvcG5nJyxcbiAgLy8gICAgIHNpemVzOiAnMTZ4MTYnLFxuICAvLyAgICAgaHJlZjogYC9pbWFnZXMvaWNvbnMvZmF2aWNvbi0xNngxNi5wbmdgLFxuICAvLyAgIH0sXG4gIC8vIF0sXG4gIC8vIFtcbiAgLy8gICAnbGluaycsXG4gIC8vICAge1xuICAvLyAgICAgcmVsOiAnaWNvbicsXG4gIC8vICAgICB0eXBlOiAnaW1hZ2UvcG5nJyxcbiAgLy8gICAgIHNpemVzOiAnMzJ4MzInLFxuICAvLyAgICAgaHJlZjogYC9pbWFnZXMvaWNvbnMvZmF2aWNvbi0zMngzMi5wbmdgLFxuICAvLyAgIH0sXG4gIC8vIF0sXG4gIFsnbGluaycsIHsgcmVsOiAnbWFuaWZlc3QnLCBocmVmOiAnL21hbmlmZXN0LndlYm1hbmlmZXN0JyB9XSxcbiAgWydtZXRhJywgeyBuYW1lOiAnYXBwbGljYXRpb24tbmFtZScsIGNvbnRlbnQ6ICdFeGFtcGxlJyB9XSxcbiAgWydtZXRhJywgeyBuYW1lOiAnYXBwbGUtbW9iaWxlLXdlYi1hcHAtdGl0bGUnLCBjb250ZW50OiAnRXhhbXBsZScgfV0sXG4gIFsnbWV0YScsIHsgbmFtZTogJ2FwcGxlLW1vYmlsZS13ZWItYXBwLXN0YXR1cy1iYXItc3R5bGUnLCBjb250ZW50OiAnYmxhY2snIH1dLFxuICAvLyBbXG4gIC8vICAgJ2xpbmsnLFxuICAvLyAgIHsgcmVsOiAnYXBwbGUtdG91Y2gtaWNvbicsIGhyZWY6IGAvaW1hZ2VzL2ljb25zL2FwcGxlLXRvdWNoLWljb24ucG5nYCB9LFxuICAvLyBdLFxuICAvLyBbXG4gIC8vICAgJ2xpbmsnLFxuICAvLyAgIHtcbiAgLy8gICAgIHJlbDogJ21hc2staWNvbicsXG4gIC8vICAgICBocmVmOiAnL2ltYWdlcy9pY29ucy9zYWZhcmktcGlubmVkLXRhYi5zdmcnLFxuICAvLyAgICAgY29sb3I6ICcjM2VhZjdjJyxcbiAgLy8gICB9LFxuICAvLyBdLFxuICBbJ21ldGEnLCB7IG5hbWU6ICdtc2FwcGxpY2F0aW9uLVRpbGVDb2xvcicsIGNvbnRlbnQ6ICcjM2VhZjdjJyB9XSxcbiAgWydtZXRhJywgeyBuYW1lOiAndGhlbWUtY29sb3InLCBjb250ZW50OiAnIzNlYWY3YycgfV0sXG5dXG4iLCAiY29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2Rpcm5hbWUgPSBcIi9oZGQvZG9jL25vdGVib29rL2RvY3MvLnZ1ZXByZXNzL2NvbmZpZ3NcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfZmlsZW5hbWUgPSBcIi9oZGQvZG9jL25vdGVib29rL2RvY3MvLnZ1ZXByZXNzL2NvbmZpZ3MvbWV0YS50c1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9pbXBvcnRfbWV0YV91cmwgPSBcImZpbGU6Ly8vaGRkL2RvYy9ub3RlYm9vay9kb2NzLy52dWVwcmVzcy9jb25maWdzL21ldGEudHNcIjtpbXBvcnQgeyBjcmVhdGVSZXF1aXJlIH0gZnJvbSAnbm9kZTptb2R1bGUnXG5pbXBvcnQgeyBmcyB9IGZyb20gJ3Z1ZXByZXNzL3V0aWxzJ1xuXG5jb25zdCByZXF1aXJlID0gY3JlYXRlUmVxdWlyZShpbXBvcnQubWV0YS51cmwpXG5cbmV4cG9ydCBjb25zdCB2ZXJzaW9uID0gZnMucmVhZEpzb25TeW5jKFxuICByZXF1aXJlLnJlc29sdmUoJ3Z1ZXByZXNzL3BhY2thZ2UuanNvbicpLFxuKS52ZXJzaW9uXG4iLCAiY29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2Rpcm5hbWUgPSBcIi9oZGQvZG9jL25vdGVib29rL2RvY3MvLnZ1ZXByZXNzL2NvbmZpZ3MvbmF2YmFyXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ZpbGVuYW1lID0gXCIvaGRkL2RvYy9ub3RlYm9vay9kb2NzLy52dWVwcmVzcy9jb25maWdzL25hdmJhci9lbi50c1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9pbXBvcnRfbWV0YV91cmwgPSBcImZpbGU6Ly8vaGRkL2RvYy9ub3RlYm9vay9kb2NzLy52dWVwcmVzcy9jb25maWdzL25hdmJhci9lbi50c1wiO2ltcG9ydCB0eXBlIHsgTmF2YmFyQ29uZmlnIH0gZnJvbSAnQHZ1ZXByZXNzL3RoZW1lLWRlZmF1bHQnXG5pbXBvcnQgeyB2ZXJzaW9uIH0gZnJvbSAnLi4vbWV0YS5qcydcblxuZXhwb3J0IGNvbnN0IG5hdmJhckVuOiBOYXZiYXJDb25maWcgPSBbXG4gIHtcbiAgICB0ZXh0OiAnR3VpZGUnLFxuICAgIGNoaWxkcmVuOiBbXG4gICAgICAnL2d1aWRlL2ludHJvZHVjdGlvbi5tZCcsXG4gICAgICAnL2d1aWRlL2dldHRpbmctc3RhcnRlZC5tZCcsXG4gICAgICAnL2d1aWRlL2NvbmZpZ3VyYXRpb24ubWQnLFxuICAgICAgJy9ndWlkZS9wYWdlLm1kJyxcbiAgICAgICcvZ3VpZGUvbWFya2Rvd24ubWQnLFxuICAgICAgJy9ndWlkZS9hc3NldHMubWQnLFxuICAgICAgJy9ndWlkZS9pMThuLm1kJyxcbiAgICAgICcvZ3VpZGUvZGVwbG95bWVudC5tZCcsXG4gICAgICAnL2d1aWRlL3RoZW1lLm1kJyxcbiAgICAgICcvZ3VpZGUvcGx1Z2luLm1kJyxcbiAgICAgICcvZ3VpZGUvYnVuZGxlci5tZCcsXG4gICAgICAnL2d1aWRlL21pZ3JhdGlvbi5tZCcsXG4gICAgICAnL2d1aWRlL3Ryb3VibGVzaG9vdGluZy5tZCcsXG4gICAgXSxcbiAgfSxcbiAge1xuICAgIHRleHQ6ICdSZWZlcmVuY2UnLFxuICAgIGNoaWxkcmVuOiBbXG4gICAgICB7XG4gICAgICAgIHRleHQ6ICdWdWVQcmVzcycsXG4gICAgICAgIGNoaWxkcmVuOiBbXG4gICAgICAgICAge1xuICAgICAgICAgICAgdGV4dDogJ0NMSScsXG4gICAgICAgICAgICBsaW5rOiAnL3JlZmVyZW5jZS9jbGkuaHRtbCcsXG4gICAgICAgICAgfSxcbiAgICAgICAgICAnL3JlZmVyZW5jZS9jb25maWcubWQnLFxuICAgICAgICAgICcvcmVmZXJlbmNlL2Zyb250bWF0dGVyLm1kJyxcbiAgICAgICAgICAnL3JlZmVyZW5jZS9jb21wb25lbnRzLm1kJyxcbiAgICAgICAgICAnL3JlZmVyZW5jZS9wbHVnaW4tYXBpLm1kJyxcbiAgICAgICAgICAnL3JlZmVyZW5jZS90aGVtZS1hcGkubWQnLFxuICAgICAgICAgICcvcmVmZXJlbmNlL2NsaWVudC1hcGkubWQnLFxuICAgICAgICAgICcvcmVmZXJlbmNlL25vZGUtYXBpLm1kJyxcbiAgICAgICAgXSxcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIHRleHQ6ICdCdW5kbGVycycsXG4gICAgICAgIGNoaWxkcmVuOiBbXG4gICAgICAgICAgJy9yZWZlcmVuY2UvYnVuZGxlci92aXRlLm1kJyxcbiAgICAgICAgICAnL3JlZmVyZW5jZS9idW5kbGVyL3dlYnBhY2subWQnLFxuICAgICAgICBdLFxuICAgICAgfSxcbiAgICBdLFxuICB9LFxuXG4gIHtcbiAgICB0ZXh0OiAnTGVhcm4gTW9yZScsXG4gICAgY2hpbGRyZW46IFtcbiAgICAgIHtcbiAgICAgICAgdGV4dDogJ0FkdmFuY2VkJyxcbiAgICAgICAgY2hpbGRyZW46IFtcbiAgICAgICAgICAnL2FkdmFuY2VkL2FyY2hpdGVjdHVyZS5tZCcsXG4gICAgICAgICAgJy9hZHZhbmNlZC9wbHVnaW4ubWQnLFxuICAgICAgICAgICcvYWR2YW5jZWQvdGhlbWUubWQnLFxuICAgICAgICAgIHtcbiAgICAgICAgICAgIHRleHQ6ICdDb29rYm9vaycsXG4gICAgICAgICAgICBsaW5rOiAnL2FkdmFuY2VkL2Nvb2tib29rLycsXG4gICAgICAgICAgfSxcbiAgICAgICAgXSxcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIHRleHQ6ICdSZXNvdXJjZXMnLFxuICAgICAgICBjaGlsZHJlbjogW1xuICAgICAgICAgIHtcbiAgICAgICAgICAgIHRleHQ6ICdPZmZpY2lhbCBFY29zeXN0ZW0nLFxuICAgICAgICAgICAgbGluazogJ2h0dHBzOi8vZWNvc3lzdGVtLnZ1ZWpzLnByZXNzLycsXG4gICAgICAgICAgfSxcbiAgICAgICAgICB7XG4gICAgICAgICAgICB0ZXh0OiAnVnVlUHJlc3MgTWFya2V0UGxhY2UnLFxuICAgICAgICAgICAgbGluazogJ2h0dHBzOi8vbWFya2V0cGxhY2UudnVlanMucHJlc3MnLFxuICAgICAgICAgIH0sXG4gICAgICAgICAge1xuICAgICAgICAgICAgdGV4dDogJ0NvbnRyaWJ1dGluZyBHdWlkZScsXG4gICAgICAgICAgICBsaW5rOiAnaHR0cHM6Ly9naXRodWIuY29tL3Z1ZXByZXNzL2NvcmUvYmxvYi9tYWluL0NPTlRSSUJVVElORy5tZCcsXG4gICAgICAgICAgfSxcbiAgICAgICAgXSxcbiAgICAgIH0sXG4gICAgXSxcbiAgfSxcbiAge1xuICAgIHRleHQ6IGB2JHt2ZXJzaW9ufWAsXG4gICAgY2hpbGRyZW46IFtcbiAgICAgIHtcbiAgICAgICAgdGV4dDogJ0NoYW5nZWxvZycsXG4gICAgICAgIGxpbms6ICdodHRwczovL2dpdGh1Yi5jb20vdnVlcHJlc3MvY29yZS9ibG9iL21haW4vQ0hBTkdFTE9HLm1kJyxcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIHRleHQ6ICd2MS54JyxcbiAgICAgICAgbGluazogJ2h0dHBzOi8vdjEudnVlcHJlc3MudnVlanMub3JnJyxcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIHRleHQ6ICd2MC54JyxcbiAgICAgICAgbGluazogJ2h0dHBzOi8vdjAudnVlcHJlc3MudnVlanMub3JnJyxcbiAgICAgIH0sXG4gICAgXSxcbiAgfSxcbl1cbiJdLAogICJtYXBwaW5ncyI6ICI7QUFBd1EsU0FBUyxpQkFBQUEsc0JBQXFCO0FBQ3RTLE9BQU8sYUFBYTtBQUNwQixTQUFTLG1CQUFtQjtBQUM1QixTQUFTLHNCQUFzQjtBQUUvQixPQUFPLFNBQVM7QUFFaEIsU0FBUyxnQ0FBZ0M7QUFDekMsU0FBUyxtQkFBbUI7QUFDNUIsU0FBUyxvQkFBb0I7QUFDN0IsU0FBUyx3QkFBd0I7QUFDakMsU0FBUyxZQUFZLFlBQVk7OztBQ1QxQixJQUFNLE9BQXFCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFtQmhDLENBQUMsUUFBUSxFQUFFLEtBQUssWUFBWSxNQUFNLHdCQUF3QixDQUFDO0FBQUEsRUFDM0QsQ0FBQyxRQUFRLEVBQUUsTUFBTSxvQkFBb0IsU0FBUyxVQUFVLENBQUM7QUFBQSxFQUN6RCxDQUFDLFFBQVEsRUFBRSxNQUFNLDhCQUE4QixTQUFTLFVBQVUsQ0FBQztBQUFBLEVBQ25FLENBQUMsUUFBUSxFQUFFLE1BQU0seUNBQXlDLFNBQVMsUUFBUSxDQUFDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFhNUUsQ0FBQyxRQUFRLEVBQUUsTUFBTSwyQkFBMkIsU0FBUyxVQUFVLENBQUM7QUFBQSxFQUNoRSxDQUFDLFFBQVEsRUFBRSxNQUFNLGVBQWUsU0FBUyxVQUFVLENBQUM7QUFDdEQ7OztBQ3ZDNFIsU0FBUyxxQkFBcUI7QUFDMVQsU0FBUyxVQUFVO0FBRDhKLElBQU0sMkNBQTJDO0FBR2xPLElBQU1DLFdBQVUsY0FBYyx3Q0FBZTtBQUV0QyxJQUFNLFVBQVUsR0FBRztBQUFBLEVBQ3hCQSxTQUFRLFFBQVEsdUJBQXVCO0FBQ3pDLEVBQUU7OztBQ0pLLElBQU0sV0FBeUI7QUFBQSxFQUNwQztBQUFBLElBQ0UsTUFBTTtBQUFBLElBQ04sVUFBVTtBQUFBLE1BQ1I7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxJQUNGO0FBQUEsRUFDRjtBQUFBLEVBQ0E7QUFBQSxJQUNFLE1BQU07QUFBQSxJQUNOLFVBQVU7QUFBQSxNQUNSO0FBQUEsUUFDRSxNQUFNO0FBQUEsUUFDTixVQUFVO0FBQUEsVUFDUjtBQUFBLFlBQ0UsTUFBTTtBQUFBLFlBQ04sTUFBTTtBQUFBLFVBQ1I7QUFBQSxVQUNBO0FBQUEsVUFDQTtBQUFBLFVBQ0E7QUFBQSxVQUNBO0FBQUEsVUFDQTtBQUFBLFVBQ0E7QUFBQSxVQUNBO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBO0FBQUEsUUFDRSxNQUFNO0FBQUEsUUFDTixVQUFVO0FBQUEsVUFDUjtBQUFBLFVBQ0E7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLElBQ0Y7QUFBQSxFQUNGO0FBQUEsRUFFQTtBQUFBLElBQ0UsTUFBTTtBQUFBLElBQ04sVUFBVTtBQUFBLE1BQ1I7QUFBQSxRQUNFLE1BQU07QUFBQSxRQUNOLFVBQVU7QUFBQSxVQUNSO0FBQUEsVUFDQTtBQUFBLFVBQ0E7QUFBQSxVQUNBO0FBQUEsWUFDRSxNQUFNO0FBQUEsWUFDTixNQUFNO0FBQUEsVUFDUjtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQTtBQUFBLFFBQ0UsTUFBTTtBQUFBLFFBQ04sVUFBVTtBQUFBLFVBQ1I7QUFBQSxZQUNFLE1BQU07QUFBQSxZQUNOLE1BQU07QUFBQSxVQUNSO0FBQUEsVUFDQTtBQUFBLFlBQ0UsTUFBTTtBQUFBLFlBQ04sTUFBTTtBQUFBLFVBQ1I7QUFBQSxVQUNBO0FBQUEsWUFDRSxNQUFNO0FBQUEsWUFDTixNQUFNO0FBQUEsVUFDUjtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFBQSxFQUNBO0FBQUEsSUFDRSxNQUFNLElBQUksT0FBTztBQUFBLElBQ2pCLFVBQVU7QUFBQSxNQUNSO0FBQUEsUUFDRSxNQUFNO0FBQUEsUUFDTixNQUFNO0FBQUEsTUFDUjtBQUFBLE1BQ0E7QUFBQSxRQUNFLE1BQU07QUFBQSxRQUNOLE1BQU07QUFBQSxNQUNSO0FBQUEsTUFDQTtBQUFBLFFBQ0UsTUFBTTtBQUFBLFFBQ04sTUFBTTtBQUFBLE1BQ1I7QUFBQSxJQUNGO0FBQUEsRUFDRjtBQUNGOzs7QUhwRkEsU0FBUyxZQUFZO0FBQ3JCLFNBQVMsb0JBQW9CO0FBbkJzSSxJQUFNQyw0Q0FBMkM7QUFNcE4sSUFBTSxFQUFFLGFBQWEsSUFBSTtBQWV6QixJQUFNLFlBQVksV0FBV0EseUNBQWU7QUFDNUMsSUFBTUMsV0FBVUMsZUFBY0YseUNBQWU7QUFDN0MsSUFBTSxTQUFTLFFBQVEsSUFBSSxhQUFhO0FBRXhDLElBQU1HLFlBQVcsS0FBSyxhQUFhLGdCQUFnQixPQUFPLENBQUM7QUFDM0QsSUFBTSxZQUFZLEtBQUssYUFBYSxnQkFBZ0IsT0FBTyxDQUFDO0FBRTVELElBQU8saUJBQVEsaUJBQWlCO0FBQUE7QUFBQSxFQUU5QixNQUFNO0FBQUE7QUFBQSxFQUdOO0FBQUE7QUFBQSxFQUdBLFNBQVM7QUFBQSxJQUNQLEtBQUs7QUFBQSxNQUNILE1BQU07QUFBQSxNQUNOLE9BQU87QUFBQSxNQUNQLGFBQWE7QUFBQSxJQUNmO0FBQUEsRUFDRjtBQUFBO0FBQUEsRUFHQSxTQUNFLFFBQVEsSUFBSSxpQkFBaUIsWUFBWSxlQUFlLElBQUksWUFBWTtBQUFBO0FBQUEsRUFHMUUsT0FBTyxhQUFhO0FBQUE7QUFBQSxJQUVsQixNQUFNO0FBQUE7QUFBQSxJQUVOLFNBQVM7QUFBQTtBQUFBLElBR1QsU0FBUztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BT1AsS0FBSztBQUFBO0FBQUEsUUFFSCxRQUFRQTtBQUFBO0FBQUEsUUFFUixTQUFTO0FBQUE7QUFBQSxRQUVULGNBQWM7QUFBQSxNQUNoQjtBQUFBLElBQ0Y7QUFBQSxJQUVBLGNBQWM7QUFBQTtBQUFBLE1BRVosS0FBSztBQUFBO0FBQUEsTUFFTCxTQUFTLENBQUM7QUFBQSxJQUNaO0FBQUEsRUFDRixDQUFDO0FBQUE7QUFBQSxFQUdELFVBQVU7QUFBQSxJQUNSLFlBQVk7QUFBQSxNQUNWLGtCQUFrQixDQUFDLGVBQWU7QUFFaEMsWUFBSSxXQUFXLFdBQVcsWUFBWSxHQUFHO0FBQ3ZDLGdCQUFNLGNBQWMsV0FBVyxNQUFNLHFCQUFxQixFQUFHLENBQUM7QUFDOUQsaUJBQU8sV0FDSjtBQUFBLFlBQ0M7QUFBQSxZQUNBLEtBQUssUUFBUUYsU0FBUSxRQUFRLEdBQUcsV0FBVyxlQUFlLENBQUM7QUFBQSxVQUM3RCxFQUNDLFFBQVEsU0FBUyxPQUFPLEVBQ3hCLFFBQVEsZUFBZSxhQUFhO0FBQUEsUUFDekM7QUFDQSxlQUFPO0FBQUEsTUFDVDtBQUFBLElBQ0Y7QUFBQSxFQUNGO0FBQUEsRUFDQSxhQUFhO0FBQUEsSUFDWCxtQkFBbUI7QUFBQSxFQUNyQjtBQUFBO0FBQUEsRUFFQSxTQUFTO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBU1AsYUFBYTtBQUFBO0FBQUEsTUFFWCxnQkFBZ0I7QUFBQSxNQUNoQixTQUFTLENBQUMsS0FBSyxHQUFHO0FBQUEsTUFDbEIsU0FBUztBQUFBLFFBQ1AsS0FBSztBQUFBLFVBQ0gsYUFBYTtBQUFBLFFBQ2Y7QUFBQSxNQUNGO0FBQUEsSUFDRixDQUFDO0FBQUEsSUFDRCx5QkFBeUI7QUFBQSxNQUN2QixlQUFlLEtBQUssUUFBUSxXQUFXLGNBQWM7QUFBQSxJQUN2RCxDQUFDO0FBQUE7QUFBQSxJQUVELFNBQ0ksWUFBWTtBQUFBLE1BQ1YsT0FBTyxDQUFDLFFBQVEsUUFBUSxRQUFRLE1BQU0sTUFBTSxLQUFLO0FBQUEsTUFDakQsT0FBTztBQUFBLElBQ1QsQ0FBQyxJQUNELENBQUM7QUFBQSxFQUNQO0FBQ0YsQ0FBQzsiLAogICJuYW1lcyI6IFsiY3JlYXRlUmVxdWlyZSIsICJyZXF1aXJlIiwgIl9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9pbXBvcnRfbWV0YV91cmwiLCAicmVxdWlyZSIsICJjcmVhdGVSZXF1aXJlIiwgIm5hdmJhckVuIl0KfQo=
