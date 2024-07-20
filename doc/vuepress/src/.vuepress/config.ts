import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/",

  lang: "en-US",
  description: "A documentation for ESPnet",

  theme,

  // Enable it with pwa
  // shouldPrefetch: false,
});
