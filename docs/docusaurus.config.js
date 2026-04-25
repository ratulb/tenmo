const config = {
  title: 'Tenmo',
  tagline: 'A lean tensor library and neural network framework built in Mojo',
  url: 'https://ratulb.github.io',
  baseUrl: '/tenmo/',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'ratulb',
  projectName: 'tenmo',
  githubUrl: 'https://github.com/ratulb/tenmo',
  trailingSlash: false,
  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          docItemComponent: '@theme/DocItem',
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/ratulb/tenmo/edit/development/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],
  plugins: [],
  themeConfig: {
    navbar: {
      title: 'Tenmo',
      logo: {
        alt: 'Tenmo Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'doc',
          docId: 'index',
          position: 'left',
          label: 'API Reference',
        },
        {
          href: 'https://github.com/ratulb/tenmo',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      copyright: `Built with Docusaurus. Mojo™ is a trademark of Modular, Inc.`,
    },
    prism: {
      languageExtensions: [],
      theme: require('prism-react-renderer').themes.github,
      darkTheme: require('prism-react-renderer').themes.dracula,
    },
  },
};

module.exports = config;