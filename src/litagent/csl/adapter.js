/* Local-only CSL adapter. Input is data; style/locale paths are allowlisted. */
const fs = require('node:fs');
const path = require('node:path');
const CSL = require('./citeproc.js');
const styles = new Set(['ieee', 'apa7', 'gbt7714_2025_numeric', 'gbt7714_2025_author_date']);
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
if (!styles.has(input.style)) throw new Error('unsupported_citation_style');
const items = new Map(input.items.map(item => [item.id, item]));
const engine = new CSL.Engine({
  retrieveLocale: lang => fs.readFileSync(path.join(__dirname, lang.startsWith('zh') ? 'locales-zh-CN.xml' : 'locales-en-US.xml'), 'utf8'),
  retrieveItem: id => { if (!items.has(id)) throw new Error('unknown_citation_item'); return items.get(id); },
}, fs.readFileSync(path.join(__dirname, input.style + '.csl'), 'utf8'));
engine.setOutputFormat('html');
const citations = [];
for (const cluster of input.clusters) {
  const updates = engine.appendCitationCluster({citationItems: cluster.map(id => ({id})), properties: {noteIndex: 0}});
  for (const [index, label] of updates) citations[index] = label;
}
const bibliography = engine.makeBibliography();
process.stdout.write(JSON.stringify({citations, bibliography: bibliography ? bibliography[1] : [], item_ids: bibliography ? bibliography[0].entry_ids.map(ids => ids[0]) : []}));
