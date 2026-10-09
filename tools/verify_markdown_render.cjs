/* Optional browser QA for the local review pages; never publishes documents. */
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.TUTORIAL_PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');
const review = path.join(root, 'outputs', 'review');

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.TUTORIAL_BROWSER_PATH});
  const page = await browser.newPage({viewport: {width: 1280, height: 1050}});
  const result = {renderer: 'Local CommonMark + MathJax in Chromium; not live GitHub', pages: [], errors: []};
  for (const file of fs.readdirSync(review).filter(n => n.endsWith('.html'))) {
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(pathToFileURL(path.join(review, file)).href);
    await page.waitForFunction(() => !!window.MathJax?.startup?.promise);
    await page.evaluate(() => window.MathJax.startup.promise);
    const data = await page.evaluate(() => ({
      math: document.querySelectorAll('mjx-container').length,
      mathErrors: [...document.querySelectorAll('[data-mjx-error]')].map(n => n.getAttribute('data-mjx-error')),
      missingImages: [...document.images].filter(i => !i.complete || i.naturalWidth === 0).map(i => i.src),
      rules: document.querySelectorAll('hr').length,
      contentOverflow: document.documentElement.scrollWidth > innerWidth + 2,
    }));
    result.pages.push({file, ...data});
    result.errors.push(...errors.map(e => ({file, error:e})), ...data.mathErrors.map(e => ({file,error:e})));
    if (data.missingImages.length || data.rules || data.contentOverflow) result.errors.push({file, error:data});
    if (file.startsWith('09-')) {
      await page.screenshot({path:path.join(review,'09-desktop.png')});
      await page.locator('h2').filter({hasText:'9.10'}).scrollIntoViewIfNeeded();
      await page.screenshot({path:path.join(review,'09-backprop.png')});
      await page.setViewportSize({width:390,height:844});
      await page.reload(); await page.evaluate(() => window.MathJax.startup.promise);
      await page.screenshot({path:path.join(review,'09-mobile.png')});
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2);
      if (overflow) result.errors.push({file,error:'Mobile horizontal page overflow'});
      await page.setViewportSize({width:1280,height:1050});
    }
    if (/^(14|18|19)-/.test(file)) {
      const number = file.slice(0,2);
      await page.screenshot({path:path.join(review,number+'-desktop.png')});
      await page.setViewportSize({width:390,height:844});
      await page.reload(); await page.evaluate(() => window.MathJax.startup.promise);
      await page.screenshot({path:path.join(review,number+'-mobile.png')});
      if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2)) {
        result.errors.push({file,error:'Mobile horizontal page overflow'});
      }
      await page.setViewportSize({width:1280,height:1050});
    }
    page.removeAllListeners('pageerror');
  }
  await browser.close();
  fs.writeFileSync(path.join(review,'browser-report.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result,null,2));
  if (result.errors.length) process.exitCode = 1;
})().catch(error => {console.error(error); process.exitCode = 1;});
