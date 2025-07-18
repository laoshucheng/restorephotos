/** @type {import('next').NextConfig} */
module.exports = {
  reactStrictMode: true,
  images: {
    domains: ['upcdn.io', 'replicate.delivery', 'lh3.googleusercontent.com'],
    unoptimized: true,
  },
  async redirects() {
    return [
      {
        source: '/github',
        destination: 'https://github.com/laoshucheng/restorephotos',
        permanent: false,
      },
      {
        source: '/deploy',
        destination: 'https://vercel.com/templates/next.js/ai-photo-restorer',
        permanent: false,
      },
    ];
  },
  webpack: (config) => {
    config.output.chunkFilename = `[name]-[chunkhash]-${Date.now()}.js`;
    return config;
  },
};
