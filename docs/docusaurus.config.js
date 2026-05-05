const {themes} = require('prism-react-renderer');

const config = {
  title: 'Inlay',
  tagline: 'Static, structural dependency injection for Python',
  url: 'https://alexdrydew.github.io',
  baseUrl: '/inlay/',
  organizationName: 'alexdrydew',
  projectName: 'inlay',
  trailingSlash: false,
  markdown: {
    remarkRehypeOptions: {
      footnoteBackContent: '\u21a9\ufe0e',
    },
  },
  presets: [
    [
      'classic',
      {
        docs: {
          path: 'content',
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          breadcrumbs: false,
        },
        blog: false,
      },
    ],
  ],
  themeConfig: {
    prism: {
      theme: themes.github,
      darkTheme: themes.palenight,
    },
  },
};

module.exports = config;
