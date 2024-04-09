// docs/.vuepress/config.ts
import { createRequire as createRequire2 } from "node:module";
import process from "node:process";
import { viteBundler } from "@vuepress/bundler-vite";
import { webpackBundler } from "@vuepress/bundler-webpack";
import { registerComponentsPlugin } from "@vuepress/plugin-register-components";
import { shikiPlugin } from "@vuepress/plugin-shiki";
import { defaultTheme } from "@vuepress/theme-default";
import { defineUserConfig } from "vuepress";
import { getDirname, path } from "vuepress/utils";
import searchPlugin from "@vuepress/plugin-search";

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
    searchPlugin(),
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
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiZG9jcy8udnVlcHJlc3MvY29uZmlnLnRzIiwgImRvY3MvLnZ1ZXByZXNzL2NvbmZpZ3MvaGVhZC50cyIsICJkb2NzLy52dWVwcmVzcy9jb25maWdzL21ldGEudHMiLCAiZG9jcy8udnVlcHJlc3MvY29uZmlncy9uYXZiYXIvZW4udHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCIvaGRkL2RvYy9ub3RlYm9vay9kb2NzLy52dWVwcmVzc1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiL2hkZC9kb2Mvbm90ZWJvb2svZG9jcy8udnVlcHJlc3MvY29uZmlnLnRzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9oZGQvZG9jL25vdGVib29rL2RvY3MvLnZ1ZXByZXNzL2NvbmZpZy50c1wiO2ltcG9ydCB7IGNyZWF0ZVJlcXVpcmUgfSBmcm9tICdub2RlOm1vZHVsZSdcbmltcG9ydCBwcm9jZXNzIGZyb20gJ25vZGU6cHJvY2VzcydcbmltcG9ydCB7IHZpdGVCdW5kbGVyIH0gZnJvbSAnQHZ1ZXByZXNzL2J1bmRsZXItdml0ZSdcbmltcG9ydCB7IHdlYnBhY2tCdW5kbGVyIH0gZnJvbSAnQHZ1ZXByZXNzL2J1bmRsZXItd2VicGFjaydcbi8vIGltcG9ydCB7IGRvY3NlYXJjaFBsdWdpbiB9IGZyb20gJ0B2dWVwcmVzcy9wbHVnaW4tZG9jc2VhcmNoJ1xuaW1wb3J0IHsgcmVnaXN0ZXJDb21wb25lbnRzUGx1Z2luIH0gZnJvbSAnQHZ1ZXByZXNzL3BsdWdpbi1yZWdpc3Rlci1jb21wb25lbnRzJ1xuaW1wb3J0IHsgc2hpa2lQbHVnaW4gfSBmcm9tICdAdnVlcHJlc3MvcGx1Z2luLXNoaWtpJ1xuaW1wb3J0IHsgZGVmYXVsdFRoZW1lIH0gZnJvbSAnQHZ1ZXByZXNzL3RoZW1lLWRlZmF1bHQnXG5pbXBvcnQgeyBkZWZpbmVVc2VyQ29uZmlnIH0gZnJvbSAndnVlcHJlc3MnXG5pbXBvcnQgeyBnZXREaXJuYW1lLCBwYXRoIH0gZnJvbSAndnVlcHJlc3MvdXRpbHMnXG5pbXBvcnQgc2VhcmNoUGx1Z2luIGZyb20gJ0B2dWVwcmVzcy9wbHVnaW4tc2VhcmNoJ1xuaW1wb3J0IHtcbiAgaGVhZCxcbiAgLy8gbmF2YmFyRW4sXG4gIC8vIHNpZGViYXJFbixcbn0gZnJvbSAnLi9jb25maWdzL2luZGV4LmpzJ1xuaW1wb3J0IHsgbG9hZCB9IGZyb20gJ2pzLXlhbWwnXG5pbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcydcblxuY29uc3QgX19kaXJuYW1lID0gZ2V0RGlybmFtZShpbXBvcnQubWV0YS51cmwpXG5jb25zdCByZXF1aXJlID0gY3JlYXRlUmVxdWlyZShpbXBvcnQubWV0YS51cmwpXG5jb25zdCBpc1Byb2QgPSBwcm9jZXNzLmVudi5OT0RFX0VOViA9PT0gJ3Byb2R1Y3Rpb24nXG5cbmNvbnN0IG5hdmJhckVuID0gbG9hZChyZWFkRmlsZVN5bmMoJ21lbnViYXJzLnltbCcsICd1dGYtOCcpKVxuY29uc3Qgc2lkZWJhckVuID0gbG9hZChyZWFkRmlsZVN5bmMoJ3NpZGViYXJzLnltbCcsICd1dGYtOCcpKVxuXG5leHBvcnQgZGVmYXVsdCBkZWZpbmVVc2VyQ29uZmlnKHtcbiAgLy8gc2V0IHNpdGUgYmFzZSB0byBkZWZhdWx0IHZhbHVlXG4gIGJhc2U6ICcvJyxcblxuICAvLyBleHRyYSB0YWdzIGluIGA8aGVhZD5gXG4gIGhlYWQsXG5cbiAgLy8gc2l0ZS1sZXZlbCBsb2NhbGVzIGNvbmZpZ1xuICBsb2NhbGVzOiB7XG4gICAgJy8nOiB7XG4gICAgICBsYW5nOiAnZW4tVVMnLFxuICAgICAgdGl0bGU6ICdFU1BuZXQtRXhhbXBsZScsXG4gICAgICBkZXNjcmlwdGlvbjogJ0V4YW1wbGUgbm90ZWJvb2tzIGZvciBFU1BuZXQnLFxuICAgIH0sXG4gIH0sXG5cbiAgLy8gc3BlY2lmeSBidW5kbGVyIHZpYSBlbnZpcm9ubWVudCB2YXJpYWJsZVxuICBidW5kbGVyOlxuICAgIHByb2Nlc3MuZW52LkRPQ1NfQlVORExFUiA9PT0gJ3dlYnBhY2snID8gd2VicGFja0J1bmRsZXIoKSA6IHZpdGVCdW5kbGVyKCksXG5cbiAgLy8gY29uZmlndXJlIGRlZmF1bHQgdGhlbWVcbiAgdGhlbWU6IGRlZmF1bHRUaGVtZSh7XG4gICAgLy8gaG9zdG5hbWU6ICdodHRwczovL3YyLnZ1ZXByZXNzLnZ1ZWpzLm9yZycsXG4gICAgbG9nbzogJy9pbWFnZXMvZXNwbmV0X2xvZ28xLnBuZycsXG4gICAgLy8gcmVwbzogJ3Z1ZXByZXNzL2RvY3MnLFxuICAgIGRvY3NEaXI6ICdkb2NzJyxcblxuICAgIC8vIHRoZW1lLWxldmVsIGxvY2FsZXMgY29uZmlnXG4gICAgbG9jYWxlczoge1xuICAgICAgLyoqXG4gICAgICAgKiBFbmdsaXNoIGxvY2FsZSBjb25maWdcbiAgICAgICAqXG4gICAgICAgKiBBcyB0aGUgZGVmYXVsdCBsb2NhbGUgb2YgQHZ1ZXByZXNzL3RoZW1lLWRlZmF1bHQgaXMgRW5nbGlzaCxcbiAgICAgICAqIHdlIGRvbid0IG5lZWQgdG8gc2V0IGFsbCBvZiB0aGUgbG9jYWxlIGZpZWxkc1xuICAgICAgICovXG4gICAgICAnLyc6IHtcbiAgICAgICAgLy8gbmF2YmFyXG4gICAgICAgIG5hdmJhcjogbmF2YmFyRW4sXG4gICAgICAgIC8vIHNpZGViYXJcbiAgICAgICAgc2lkZWJhcjogc2lkZWJhckVuLFxuICAgICAgICAvLyBwYWdlIG1ldGFcbiAgICAgICAgZWRpdExpbmtUZXh0OiAnRWRpdCB0aGlzIHBhZ2Ugb24gR2l0SHViJyxcbiAgICAgIH0sXG4gICAgfSxcblxuICAgIHRoZW1lUGx1Z2luczoge1xuICAgICAgLy8gb25seSBlbmFibGUgZ2l0IHBsdWdpbiBpbiBwcm9kdWN0aW9uIG1vZGVcbiAgICAgIGdpdDogaXNQcm9kLFxuICAgICAgLy8gdXNlIHNoaWtpIHBsdWdpbiBpbiBwcm9kdWN0aW9uIG1vZGUgaW5zdGVhZFxuICAgICAgcHJpc21qczogIWlzUHJvZCxcbiAgICB9LFxuICB9KSxcblxuICAvLyBjb25maWd1cmUgbWFya2Rvd25cbiAgbWFya2Rvd246IHtcbiAgICBpbXBvcnRDb2RlOiB7XG4gICAgICBoYW5kbGVJbXBvcnRQYXRoOiAoaW1wb3J0UGF0aCkgPT4ge1xuICAgICAgICAvLyBoYW5kbGUgQHZ1ZXByZXNzIHBhY2thZ2VzIGltcG9ydCBwYXRoXG4gICAgICAgIGlmIChpbXBvcnRQYXRoLnN0YXJ0c1dpdGgoJ0B2dWVwcmVzcy8nKSkge1xuICAgICAgICAgIGNvbnN0IHBhY2thZ2VOYW1lID0gaW1wb3J0UGF0aC5tYXRjaCgvXihAdnVlcHJlc3NcXC9bXi9dKikvKSFbMV1cbiAgICAgICAgICByZXR1cm4gaW1wb3J0UGF0aFxuICAgICAgICAgICAgLnJlcGxhY2UoXG4gICAgICAgICAgICAgIHBhY2thZ2VOYW1lLFxuICAgICAgICAgICAgICBwYXRoLmRpcm5hbWUocmVxdWlyZS5yZXNvbHZlKGAke3BhY2thZ2VOYW1lfS9wYWNrYWdlLmpzb25gKSksXG4gICAgICAgICAgICApXG4gICAgICAgICAgICAucmVwbGFjZSgnL3NyYy8nLCAnL2xpYi8nKVxuICAgICAgICAgICAgLnJlcGxhY2UoL2hvdEtleVxcLnRzJC8sICdob3RLZXkuZC50cycpXG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGltcG9ydFBhdGhcbiAgICAgIH0sXG4gICAgfSxcbiAgfSxcbiAgdGhlbWVDb25maWc6IHtcbiAgICBzZWFyY2hQbGFjZWhvbGRlcjogJ1NlYXJjaC4uLidcbiAgfSxcbiAgLy8gdXNlIHBsdWdpbnNcbiAgcGx1Z2luczogW1xuICAgIC8vIGRvY3NlYXJjaFBsdWdpbih7XG4gICAgLy8gICBhcHBJZDogJycsXG4gICAgLy8gICBhcGlLZXk6ICcnLFxuICAgIC8vICAgaW5kZXhOYW1lOiAndnVlcHJlc3MnLFxuICAgIC8vICAgc2VhcmNoUGFyYW1ldGVyczoge1xuICAgIC8vICAgICBmYWNldEZpbHRlcnM6IFsndGFnczp2MiddLFxuICAgIC8vICAgfSxcbiAgICAvLyB9KSxcbiAgICBzZWFyY2hQbHVnaW4oKSxcbiAgICByZWdpc3RlckNvbXBvbmVudHNQbHVnaW4oe1xuICAgICAgY29tcG9uZW50c0RpcjogcGF0aC5yZXNvbHZlKF9fZGlybmFtZSwgJy4vY29tcG9uZW50cycpLFxuICAgIH0pLFxuICAgIC8vIG9ubHkgZW5hYmxlIHNoaWtpIHBsdWdpbiBpbiBwcm9kdWN0aW9uIG1vZGVcbiAgICBpc1Byb2RcbiAgICAgID8gc2hpa2lQbHVnaW4oe1xuICAgICAgICAgIGxhbmdzOiBbJ2Jhc2gnLCAnZGlmZicsICdqc29uJywgJ21kJywgJ3RzJywgJ3Z1ZSddLFxuICAgICAgICAgIHRoZW1lOiAnZGFyay1wbHVzJyxcbiAgICAgICAgfSlcbiAgICAgIDogW10sXG4gIF0sXG59KVxuIiwgImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCIvaGRkL2RvYy9ub3RlYm9vay9kb2NzLy52dWVwcmVzcy9jb25maWdzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ZpbGVuYW1lID0gXCIvaGRkL2RvYy9ub3RlYm9vay9kb2NzLy52dWVwcmVzcy9jb25maWdzL2hlYWQudHNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL2hkZC9kb2Mvbm90ZWJvb2svZG9jcy8udnVlcHJlc3MvY29uZmlncy9oZWFkLnRzXCI7aW1wb3J0IHR5cGUgeyBIZWFkQ29uZmlnIH0gZnJvbSAndnVlcHJlc3MvY29yZSdcblxuZXhwb3J0IGNvbnN0IGhlYWQ6IEhlYWRDb25maWdbXSA9IFtcbiAgLy8gW1xuICAvLyAgICdsaW5rJyxcbiAgLy8gICB7XG4gIC8vICAgICByZWw6ICdpY29uJyxcbiAgLy8gICAgIHR5cGU6ICdpbWFnZS9wbmcnLFxuICAvLyAgICAgc2l6ZXM6ICcxNngxNicsXG4gIC8vICAgICBocmVmOiBgL2ltYWdlcy9pY29ucy9mYXZpY29uLTE2eDE2LnBuZ2AsXG4gIC8vICAgfSxcbiAgLy8gXSxcbiAgLy8gW1xuICAvLyAgICdsaW5rJyxcbiAgLy8gICB7XG4gIC8vICAgICByZWw6ICdpY29uJyxcbiAgLy8gICAgIHR5cGU6ICdpbWFnZS9wbmcnLFxuICAvLyAgICAgc2l6ZXM6ICczMngzMicsXG4gIC8vICAgICBocmVmOiBgL2ltYWdlcy9pY29ucy9mYXZpY29uLTMyeDMyLnBuZ2AsXG4gIC8vICAgfSxcbiAgLy8gXSxcbiAgWydsaW5rJywgeyByZWw6ICdtYW5pZmVzdCcsIGhyZWY6ICcvbWFuaWZlc3Qud2VibWFuaWZlc3QnIH1dLFxuICBbJ21ldGEnLCB7IG5hbWU6ICdhcHBsaWNhdGlvbi1uYW1lJywgY29udGVudDogJ0V4YW1wbGUnIH1dLFxuICBbJ21ldGEnLCB7IG5hbWU6ICdhcHBsZS1tb2JpbGUtd2ViLWFwcC10aXRsZScsIGNvbnRlbnQ6ICdFeGFtcGxlJyB9XSxcbiAgWydtZXRhJywgeyBuYW1lOiAnYXBwbGUtbW9iaWxlLXdlYi1hcHAtc3RhdHVzLWJhci1zdHlsZScsIGNvbnRlbnQ6ICdibGFjaycgfV0sXG4gIC8vIFtcbiAgLy8gICAnbGluaycsXG4gIC8vICAgeyByZWw6ICdhcHBsZS10b3VjaC1pY29uJywgaHJlZjogYC9pbWFnZXMvaWNvbnMvYXBwbGUtdG91Y2gtaWNvbi5wbmdgIH0sXG4gIC8vIF0sXG4gIC8vIFtcbiAgLy8gICAnbGluaycsXG4gIC8vICAge1xuICAvLyAgICAgcmVsOiAnbWFzay1pY29uJyxcbiAgLy8gICAgIGhyZWY6ICcvaW1hZ2VzL2ljb25zL3NhZmFyaS1waW5uZWQtdGFiLnN2ZycsXG4gIC8vICAgICBjb2xvcjogJyMzZWFmN2MnLFxuICAvLyAgIH0sXG4gIC8vIF0sXG4gIFsnbWV0YScsIHsgbmFtZTogJ21zYXBwbGljYXRpb24tVGlsZUNvbG9yJywgY29udGVudDogJyMzZWFmN2MnIH1dLFxuICBbJ21ldGEnLCB7IG5hbWU6ICd0aGVtZS1jb2xvcicsIGNvbnRlbnQ6ICcjM2VhZjdjJyB9XSxcbl1cbiIsICJjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfZGlybmFtZSA9IFwiL2hkZC9kb2Mvbm90ZWJvb2svZG9jcy8udnVlcHJlc3MvY29uZmlnc1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiL2hkZC9kb2Mvbm90ZWJvb2svZG9jcy8udnVlcHJlc3MvY29uZmlncy9tZXRhLnRzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9oZGQvZG9jL25vdGVib29rL2RvY3MvLnZ1ZXByZXNzL2NvbmZpZ3MvbWV0YS50c1wiO2ltcG9ydCB7IGNyZWF0ZVJlcXVpcmUgfSBmcm9tICdub2RlOm1vZHVsZSdcbmltcG9ydCB7IGZzIH0gZnJvbSAndnVlcHJlc3MvdXRpbHMnXG5cbmNvbnN0IHJlcXVpcmUgPSBjcmVhdGVSZXF1aXJlKGltcG9ydC5tZXRhLnVybClcblxuZXhwb3J0IGNvbnN0IHZlcnNpb24gPSBmcy5yZWFkSnNvblN5bmMoXG4gIHJlcXVpcmUucmVzb2x2ZSgndnVlcHJlc3MvcGFja2FnZS5qc29uJyksXG4pLnZlcnNpb25cbiIsICJjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfZGlybmFtZSA9IFwiL2hkZC9kb2Mvbm90ZWJvb2svZG9jcy8udnVlcHJlc3MvY29uZmlncy9uYXZiYXJcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfZmlsZW5hbWUgPSBcIi9oZGQvZG9jL25vdGVib29rL2RvY3MvLnZ1ZXByZXNzL2NvbmZpZ3MvbmF2YmFyL2VuLnRzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9oZGQvZG9jL25vdGVib29rL2RvY3MvLnZ1ZXByZXNzL2NvbmZpZ3MvbmF2YmFyL2VuLnRzXCI7aW1wb3J0IHR5cGUgeyBOYXZiYXJDb25maWcgfSBmcm9tICdAdnVlcHJlc3MvdGhlbWUtZGVmYXVsdCdcbmltcG9ydCB7IHZlcnNpb24gfSBmcm9tICcuLi9tZXRhLmpzJ1xuXG5leHBvcnQgY29uc3QgbmF2YmFyRW46IE5hdmJhckNvbmZpZyA9IFtcbiAge1xuICAgIHRleHQ6ICdHdWlkZScsXG4gICAgY2hpbGRyZW46IFtcbiAgICAgICcvZ3VpZGUvaW50cm9kdWN0aW9uLm1kJyxcbiAgICAgICcvZ3VpZGUvZ2V0dGluZy1zdGFydGVkLm1kJyxcbiAgICAgICcvZ3VpZGUvY29uZmlndXJhdGlvbi5tZCcsXG4gICAgICAnL2d1aWRlL3BhZ2UubWQnLFxuICAgICAgJy9ndWlkZS9tYXJrZG93bi5tZCcsXG4gICAgICAnL2d1aWRlL2Fzc2V0cy5tZCcsXG4gICAgICAnL2d1aWRlL2kxOG4ubWQnLFxuICAgICAgJy9ndWlkZS9kZXBsb3ltZW50Lm1kJyxcbiAgICAgICcvZ3VpZGUvdGhlbWUubWQnLFxuICAgICAgJy9ndWlkZS9wbHVnaW4ubWQnLFxuICAgICAgJy9ndWlkZS9idW5kbGVyLm1kJyxcbiAgICAgICcvZ3VpZGUvbWlncmF0aW9uLm1kJyxcbiAgICAgICcvZ3VpZGUvdHJvdWJsZXNob290aW5nLm1kJyxcbiAgICBdLFxuICB9LFxuICB7XG4gICAgdGV4dDogJ1JlZmVyZW5jZScsXG4gICAgY2hpbGRyZW46IFtcbiAgICAgIHtcbiAgICAgICAgdGV4dDogJ1Z1ZVByZXNzJyxcbiAgICAgICAgY2hpbGRyZW46IFtcbiAgICAgICAgICB7XG4gICAgICAgICAgICB0ZXh0OiAnQ0xJJyxcbiAgICAgICAgICAgIGxpbms6ICcvcmVmZXJlbmNlL2NsaS5odG1sJyxcbiAgICAgICAgICB9LFxuICAgICAgICAgICcvcmVmZXJlbmNlL2NvbmZpZy5tZCcsXG4gICAgICAgICAgJy9yZWZlcmVuY2UvZnJvbnRtYXR0ZXIubWQnLFxuICAgICAgICAgICcvcmVmZXJlbmNlL2NvbXBvbmVudHMubWQnLFxuICAgICAgICAgICcvcmVmZXJlbmNlL3BsdWdpbi1hcGkubWQnLFxuICAgICAgICAgICcvcmVmZXJlbmNlL3RoZW1lLWFwaS5tZCcsXG4gICAgICAgICAgJy9yZWZlcmVuY2UvY2xpZW50LWFwaS5tZCcsXG4gICAgICAgICAgJy9yZWZlcmVuY2Uvbm9kZS1hcGkubWQnLFxuICAgICAgICBdLFxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgdGV4dDogJ0J1bmRsZXJzJyxcbiAgICAgICAgY2hpbGRyZW46IFtcbiAgICAgICAgICAnL3JlZmVyZW5jZS9idW5kbGVyL3ZpdGUubWQnLFxuICAgICAgICAgICcvcmVmZXJlbmNlL2J1bmRsZXIvd2VicGFjay5tZCcsXG4gICAgICAgIF0sXG4gICAgICB9LFxuICAgIF0sXG4gIH0sXG5cbiAge1xuICAgIHRleHQ6ICdMZWFybiBNb3JlJyxcbiAgICBjaGlsZHJlbjogW1xuICAgICAge1xuICAgICAgICB0ZXh0OiAnQWR2YW5jZWQnLFxuICAgICAgICBjaGlsZHJlbjogW1xuICAgICAgICAgICcvYWR2YW5jZWQvYXJjaGl0ZWN0dXJlLm1kJyxcbiAgICAgICAgICAnL2FkdmFuY2VkL3BsdWdpbi5tZCcsXG4gICAgICAgICAgJy9hZHZhbmNlZC90aGVtZS5tZCcsXG4gICAgICAgICAge1xuICAgICAgICAgICAgdGV4dDogJ0Nvb2tib29rJyxcbiAgICAgICAgICAgIGxpbms6ICcvYWR2YW5jZWQvY29va2Jvb2svJyxcbiAgICAgICAgICB9LFxuICAgICAgICBdLFxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgdGV4dDogJ1Jlc291cmNlcycsXG4gICAgICAgIGNoaWxkcmVuOiBbXG4gICAgICAgICAge1xuICAgICAgICAgICAgdGV4dDogJ09mZmljaWFsIEVjb3N5c3RlbScsXG4gICAgICAgICAgICBsaW5rOiAnaHR0cHM6Ly9lY29zeXN0ZW0udnVlanMucHJlc3MvJyxcbiAgICAgICAgICB9LFxuICAgICAgICAgIHtcbiAgICAgICAgICAgIHRleHQ6ICdWdWVQcmVzcyBNYXJrZXRQbGFjZScsXG4gICAgICAgICAgICBsaW5rOiAnaHR0cHM6Ly9tYXJrZXRwbGFjZS52dWVqcy5wcmVzcycsXG4gICAgICAgICAgfSxcbiAgICAgICAgICB7XG4gICAgICAgICAgICB0ZXh0OiAnQ29udHJpYnV0aW5nIEd1aWRlJyxcbiAgICAgICAgICAgIGxpbms6ICdodHRwczovL2dpdGh1Yi5jb20vdnVlcHJlc3MvY29yZS9ibG9iL21haW4vQ09OVFJJQlVUSU5HLm1kJyxcbiAgICAgICAgICB9LFxuICAgICAgICBdLFxuICAgICAgfSxcbiAgICBdLFxuICB9LFxuICB7XG4gICAgdGV4dDogYHYke3ZlcnNpb259YCxcbiAgICBjaGlsZHJlbjogW1xuICAgICAge1xuICAgICAgICB0ZXh0OiAnQ2hhbmdlbG9nJyxcbiAgICAgICAgbGluazogJ2h0dHBzOi8vZ2l0aHViLmNvbS92dWVwcmVzcy9jb3JlL2Jsb2IvbWFpbi9DSEFOR0VMT0cubWQnLFxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgdGV4dDogJ3YxLngnLFxuICAgICAgICBsaW5rOiAnaHR0cHM6Ly92MS52dWVwcmVzcy52dWVqcy5vcmcnLFxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgdGV4dDogJ3YwLngnLFxuICAgICAgICBsaW5rOiAnaHR0cHM6Ly92MC52dWVwcmVzcy52dWVqcy5vcmcnLFxuICAgICAgfSxcbiAgICBdLFxuICB9LFxuXVxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUF3USxTQUFTLGlCQUFBQSxzQkFBcUI7QUFDdFMsT0FBTyxhQUFhO0FBQ3BCLFNBQVMsbUJBQW1CO0FBQzVCLFNBQVMsc0JBQXNCO0FBRS9CLFNBQVMsZ0NBQWdDO0FBQ3pDLFNBQVMsbUJBQW1CO0FBQzVCLFNBQVMsb0JBQW9CO0FBQzdCLFNBQVMsd0JBQXdCO0FBQ2pDLFNBQVMsWUFBWSxZQUFZO0FBQ2pDLE9BQU8sa0JBQWtCOzs7QUNSbEIsSUFBTSxPQUFxQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FB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QUM3RCxFQUNDLFFBQVEsU0FBUyxPQUFPLEVBQ3hCLFFBQVEsZUFBZSxhQUFhO0FBQUEsUUFDekM7QUFDQSxlQUFPO0FBQUEsTUFDVDtBQUFBLElBQ0Y7QUFBQSxFQUNGO0FBQUEsRUFDQSxhQUFhO0FBQUEsSUFDWCxtQkFBbUI7QUFBQSxFQUNyQjtBQUFBO0FBQUEsRUFFQSxTQUFTO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBU1AsYUFBYTtBQUFBLElBQ2IseUJBQXlCO0FBQUEsTUFDdkIsZUFBZSxLQUFLLFFBQVEsV0FBVyxjQUFjO0FBQUEsSUFDdkQsQ0FBQztBQUFBO0FBQUEsSUFFRCxTQUNJLFlBQVk7QUFBQSxNQUNWLE9BQU8sQ0FBQyxRQUFRLFFBQVEsUUFBUSxNQUFNLE1BQU0sS0FBSztBQUFBLE1BQ2pELE9BQU87QUFBQSxJQUNULENBQUMsSUFDRCxDQUFDO0FBQUEsRUFDUDtBQUNGLENBQUM7IiwKICAibmFtZXMiOiBbImNyZWF0ZVJlcXVpcmUiLCAicmVxdWlyZSIsICJfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsIiwgInJlcXVpcmUiLCAiY3JlYXRlUmVxdWlyZSIsICJuYXZiYXJFbiJdCn0K
