const config = {
  title: 'Inlay',
  tagline: 'Static, structural dependency injection for Python',
  url: 'https://alexdrydew.github.io',
  baseUrl: '/inlay/',
  organizationName: 'alexdrydew',
  projectName: 'inlay',
  trailingSlash: false,
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
};

module.exports = config;
