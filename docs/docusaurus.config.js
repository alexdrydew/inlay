const { createRosePineThemes } = require("./src/prism/rosePine");

async function createConfig() {
  const { rosePine, rosePineDawn } = await createRosePineThemes();

  return {
    title: "Inlay",
    tagline: "Static, structural dependency injection for Python",
    url: "https://alexdrydew.github.io",
    baseUrl: "/inlay/",
    organizationName: "alexdrydew",
    projectName: "inlay",
    trailingSlash: false,
    markdown: {
      remarkRehypeOptions: {
        footnoteBackContent: "\u21a9\ufe0e",
      },
    },
    presets: [
      [
        "classic",
        {
          docs: {
            path: "content",
            routeBasePath: "/",
            sidebarPath: "./sidebars.js",
            breadcrumbs: false,
          },
          blog: false,
          theme: {
            customCss: "./src/css/custom.css",
          },
        },
      ],
    ],
    themeConfig: {
      prism: {
        theme: rosePineDawn,
        darkTheme: rosePine,
      },
    },
  };
}

module.exports = createConfig;
