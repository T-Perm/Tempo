import { defineConfig } from 'vitest/config';

// Separate config for the dataset generator script (test/*.gen.js), which the
// main vitest.config.js `include` (test/**/*.test.js) deliberately excludes so
// it doesn't run as part of `npm test`. See generate-selection-dataset.gen.js
// header comment for why it must still run through Vitest.
export default defineConfig({
  test: {
    environment: 'jsdom',
    setupFiles: ['./test/setup.js'],
    include: ['test/**/*.gen.js'],
  },
});
