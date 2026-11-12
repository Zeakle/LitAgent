"""Exercise report navigation and responsive history controls without paid calls."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from litagent.report_render import render_report_projection


@pytest.mark.parametrize("label", ["摘要", "Abstract"])
def test_outline_navigates_only_unique_attached_current_report_targets(label):
    """Use host-generated anchors and the real outline/navigation consumers."""
    projection = render_report_projection(
        f"# Survey\n\n## {label}\n\nOne.\n\n## {label}\n\nTwo.\n\n"
        "## Appendix A\n\n### Queries\n\nRecords."
    )
    script = r"""
const fs = require('node:fs');
(async () => {
 let active = null;
 class Element {
   constructor(tag = 'div') {this.tagName = tag.toUpperCase();this.children = [];this.parentElement = null;this.isConnected = true;this.id = '';this.open = false;this.scrollTop = 0;this.clientTop = 2;this.events = {};this.attributes = {};this.rect = {top: 0};}
   append(...children) {for (const child of children) {child.parentElement = this;this.children.push(child);}}
   replaceChildren(...children) {this.children = [];this.append(...children);}
   addEventListener(key, callback) {this.events[key] = callback;}
   setAttribute(key, value) {this.attributes[key] = value;}
   contains(node) {return this === node || this.children.some(child => child.contains(node));}
   querySelectorAll(selector) {const descendants = this.children.flatMap(child => [child, ...child.querySelectorAll(selector)]);return selector === '[id]' ? descendants.filter(child => child.id) : descendants;}
   getBoundingClientRect() {return this.rect;}
   focus(options) {if (!options?.preventScroll) throw Error('focus moved a different scroll ancestor');active = this;}
   scrollTo(options) {this.lastScroll = options;this.scrollTop = options.top;}
 }
 global.document = {createElement: tag => new Element(tag)};
 const module = await import('data:text/javascript;base64,' + fs.readFileSync(process.argv[1]).toString('base64'));
 const projection = JSON.parse(process.argv[2]);
 const scroll = new Element(), report = new Element('article'), outline = new Element('nav');scroll.append(report);scroll.rect.top = 100;scroll.scrollTop = 900;
 const headings = projection.headings.filter(item => item.level >= 2);
 const targets = headings.map((heading, index) => {const target = new Element(`h${heading.level}`);target.id = heading.anchor;target.rect.top = 200 + index * 300;report.append(target);return target;});
 const appendix = new Element('details'), nested = new Element('details');
 report.children = report.children.filter(child => !targets.slice(-2).includes(child));report.append(appendix);appendix.append(targets.at(-2), nested);nested.append(targets.at(-1));
 const outside = new Element('h2');outside.id = headings[0].anchor;
 global.document.getElementById = () => {throw Error('global heading lookup borrowed a hidden document');};
 const outcomes = [];
 module.renderOutline(outline, projection.headings, heading => outcomes.push(module.navigateReportHeading(report, heading, scroll)));
 if (outline.children.length !== headings.length) throw Error('host headings changed');
 outline.children[1].events.click();
 if (!outcomes.at(-1) || active !== targets[1] || scroll.lastScroll.top !== 1298) throw Error('same-title section was not targeted independently');
 outline.children.at(-1).events.click();
 if (!outcomes.at(-1) || !appendix.open || !nested.open || active !== targets.at(-1)) throw Error('nested appendix was not expanded and focused');
 const before = scroll.scrollTop;
 if (module.navigateReportHeading(report, {anchor: 'unknown'}, scroll) || scroll.scrollTop !== before) throw Error('missing target reported success');
 const duplicate = new Element('h2');duplicate.id = headings[0].anchor;report.append(duplicate);
 if (module.navigateReportHeading(report, headings[0], scroll)) throw Error('ambiguous current anchor accepted');
 targets[1].isConnected = false;
 if (module.navigateReportHeading(report, headings[1], scroll)) throw Error('detached old target accepted');
 report.isConnected = false;
 if (module.navigateReportHeading(report, headings.at(-1), scroll)) throw Error('detached old report accepted');
 console.log(JSON.stringify({headings: headings.length, model_requests: 0}));
})().catch(error => {console.error(error);process.exit(1);});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo/report-view.js").resolve()),
            json.dumps(projection),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_sidebar_closes_at_desktop_boundary_and_preserves_escape_priority():
    """Run actual app handlers with only unrelated views and transport isolated."""
    script = r"""
const fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');
(async () => {
 let focused = null, dialog = null;
 class Element {
   constructor() {this.events = {};this.children = [];this.dataset = {};this.options = [];this.value = '';this.hidden = false;this.checked = false;this.disabled = false;this.textContent = '';this.isConnected = true;this.attributes = {};this.classList = {add(){}, remove(){}};}
   addEventListener(name, fn) {this.events[name] = fn;}
   replaceChildren(...children) {this.children = children;}
   append(...children) {this.children.push(...children);}
   setAttribute(name, value) {this.attributes[name] = value;}
   querySelectorAll() {return [];}
   close() {this.open = false;}
   showModal() {this.open = true;}
   add(option) {this.options.push(option);}
   focus() {focused = this;}
 }
 const elements = new Map(), $ = selector => {if (!elements.has(selector)) elements.set(selector, new Element());return elements.get(selector);};
 const documentEvents = {}, mediaEvents = {}, calls = [];
 const media = {matches: true, addEventListener: (name, callback) => {mediaEvents[name] = callback;}};
 const sandbox = vm.createContext({document: {documentElement: {lang: ''}, addEventListener: (name, callback) => {documentEvents[name] = callback;}, querySelector: selector => selector === 'dialog[open]' ? dialog : $(selector), querySelectorAll: () => [], createElement: () => new Element()},
   matchMedia: query => {if (query !== '(max-width: 760px)') throw Error('CSS/handler breakpoints differ');return media;},
   localStorage: {getItem: () => null, setItem(){}}, location: {search: '', href: ''}, history: {replaceState(){}}, URLSearchParams, AbortController,
   Option: function(label, value) {this.textContent = label;this.value = String(value);}, console, setTimeout, clearTimeout,
   fetch: async (url, options) => {calls.push({url, options});return {ok: true, json: async () => url === '/workbench/context' ? {mode: 'presentation', live_enabled: false} : []};}});
 let state = {locale: 'zh-CN', runs: [], view: null, draftReviewState: {}, activeSection: 'report'};
 const mocks = {createWorkbenchStore: () => ({get: () => state, subscribe(){}, update: patch => {state = {...state, ...patch};}}), createExecutionWorkspace: () => ({render(){}, bind(){}}),
   evidenceMap: () => new Map(), renderEvidenceList(){}, renderExecutionInspector(){}, renderHistory(){}, renderQuality(){}, locateReportDiagnostic: () => false, renderReview(){}, renderOutline(){}, renderSearchRecords(){}, renderReportDocument: () => ({headings: []}), navigateReportHeading: () => false,
   renderBenchmarkDetail(){}, renderBenchmarkList(){}, renderBenchmarkObservation(){}, adaptArchiveView(){}, adaptInteractiveView(){}, chooseDefaultSection: () => 'report', normalizeRunList: () => [], bindFormatControls(){}};
 const real = new Set(['app.js', 'audit-controls.js', 'api-client.js', 'i18n.js', 'benchmark-controller.js']), cache = new Map();
 async function load(filename) {if (cache.has(filename)) return cache.get(filename);
   const module = real.has(filename) ? new vm.SourceTextModule(fs.readFileSync(path.join(process.argv[1], filename), 'utf8'), {context: sandbox, identifier: filename}) : new vm.SyntheticModule(Object.keys(mocks), function(){for (const [name, value] of Object.entries(mocks)) this.setExport(name, value);}, {context: sandbox, identifier: filename});
   cache.set(filename, module);await module.link(specifier => load(specifier.split('?')[0].replace('./', '')));return module;}
 await (await load('app.js')).evaluate();await new Promise(resolve => setImmediate(resolve));
 const open = () => $('#sidebar-toggle').events.click(), escape = () => {let prevented = false;documentEvents.keydown({key: 'Escape', preventDefault(){prevented = true;}});return prevented;};
 const closed = () => $('#run-sidebar').dataset.open === 'false' && $('#sidebar-scrim').hidden && $('#sidebar-toggle').attributes['aria-expanded'] === 'false';
 open();if ($('#run-sidebar').dataset.open !== 'true' || $('#sidebar-scrim').hidden || $('#sidebar-toggle').attributes['aria-expanded'] !== 'true') throw Error('mobile history did not open');
 media.matches = false;mediaEvents.change({matches: false});if (!closed()) throw Error('desktop retained mobile overlay state');
 open();if (!closed()) throw Error('desktop action created an overlay');
 media.matches = true;mediaEvents.change({matches: true});if (!closed()) throw Error('return to mobile reopened history');
 open();$('#sidebar-scrim').events.click();if (!closed() || focused !== $('#sidebar-toggle')) throw Error('scrim did not close and restore focus');
 open();dialog = {};if (escape() || closed()) throw Error('sidebar intercepted dialog Escape');dialog = null;
 $('#inspector-drawer').dataset.open = 'true';if (!escape() || $('#inspector-drawer').dataset.open !== 'false' || closed()) throw Error('Inspector dismissal also closed history');
 if (!escape() || !closed() || focused !== $('#sidebar-toggle')) throw Error('history Escape did not restore focus');
 if (calls.some(call => call.options?.method === 'POST' || /\/audits/.test(call.url))) throw Error('navigation triggered paid work');
 console.log(JSON.stringify({viewport_transitions: 2, model_requests: 0}));
})().catch(error => {console.error(error);process.exit(1);});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "--experimental-vm-modules",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
