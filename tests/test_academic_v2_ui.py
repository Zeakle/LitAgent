"""Exercise real appendix rendering and source navigation without model calls."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_appendix_folding_source_navigation_and_version_bound_records(locale):
    script = r"""
const fs=require('node:fs');
(async()=>{
 class Element {
  constructor(tag='',text=''){this.tagName=tag.toUpperCase();this.nodeType=1;this.childNodes=text?[{nodeType:3,textContent:text,querySelectorAll:()=>[]}]:[];this.attrs={};this.classList={add(){},remove(){}};this.events={};}
  get textContent(){return this.childNodes.map(n=>n.textContent||'').join('');} set textContent(value){this.childNodes=value?[{nodeType:3,textContent:value,querySelectorAll:()=>[]}]:[];}
  get parentElement(){return this.parent;} get nextSibling(){const all=this.parent?.childNodes||[];return all[all.indexOf(this)+1]||null;}
  append(...nodes){for(const n of nodes){if(n.parent)n.parent.childNodes.splice(n.parent.childNodes.indexOf(n),1);n.parent=this;this.childNodes.push(n);}}
  before(n){n.parent=this.parent;this.parent.childNodes.splice(this.parent.childNodes.indexOf(this),0,n);}
  replaceChildren(...nodes){this.childNodes=[];this.append(...nodes);}getAttribute(k){return this.attrs[k]||null;}setAttribute(k,v){this.attrs[k]=v;}
  addEventListener(k,v){this.events[k]=v;}querySelector(){return null;}querySelectorAll(selector){const nodes=this.childNodes.flatMap(n=>[n,...n.querySelectorAll('*')]);if(selector==='h2')return nodes.filter(n=>n.tagName==='H2');if(selector==='*')return nodes;return [];}
  closest(selector){let n=this;while(n){if(selector==='details'&&n.tagName==='DETAILS')return n;n=n.parent;}return null;}
  scrollIntoView(){this.scrolled=true;}
 }
 global.Node={TEXT_NODE:3,ELEMENT_NODE:1};global.NodeFilter={SHOW_TEXT:4};
 global.document={createElement:tag=>new Element(tag),createDocumentFragment:()=>new Element('fragment'),createTextNode:text=>({nodeType:3,textContent:text,querySelectorAll:()=>[]}),createTreeWalker:()=>({nextNode:()=>false})};
 global.DOMParser=class{parseFromString(html){const body=new Element('body');if(html==='record')body.append(new Element('p','exact query'));else {const intro=new Element('h2','Introduction');intro.id='introduction';const refs=new Element('h2','References');refs.id='references';const appendix=new Element('h2','Appendix A');appendix.id='appendix-a';const text=new Element('p','original runtime record');text.attrs['data-source-unit-ordinal']='4';body.append(intro,new Element('p','body'),refs,new Element('p','reference'),appendix,text);}return{body};}};
 const load=async name=>import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/'+name).toString('base64'));
 const {renderReportDocument,renderSearchRecords}=await load('report-view.js');const {locateReportDiagnostic}=await load('quality-view.js');const {translator}=await load('i18n.js');const t=translator(process.argv[2]);
 const root=new Element('article');renderReportDocument(root,{html:'body',appendix_anchor:'appendix-a'},new Map(),t,()=>{});
 const details=root.childNodes.find(n=>n.tagName==='DETAILS');if(!details||details.open)throw Error('appendix not initially folded');
 if(details.childNodes[0].tagName!=='SUMMARY'||details.childNodes[0].textContent!=='Appendix A')throw Error('keyboard accessible summary missing');
 const paragraph=details.childNodes.find(n=>n.tagName==='P');if(paragraph.attrs['data-source-unit-ordinal']!=='4')throw Error('source ordinal lost in fold');
 const before=JSON.stringify(paragraph.attrs);root.querySelector=selector=>selector.includes('"4"')?paragraph:null;
 const survey='original runtime record';if(!locateReportDiagnostic(root,survey,{source_unit_id:'unit:4:0',source_start:0,source_end:survey.length,source_quote:survey})||!details.open||!paragraph.scrolled)throw Error('navigation failed to open appendix');
 if(JSON.stringify(paragraph.attrs)!==before)throw Error('navigation mutated source identity');
 const records=new Element('section');const projection={source_sha256:'a'.repeat(64),methodology:{source_sha256:'a'.repeat(64),html:'record'}};
 renderSearchRecords(records,projection,t);if(!records.childNodes.some(n=>n.tagName==='DETAILS'))throw Error('bound record not shown');
 renderSearchRecords(records,{...projection,source_sha256:'b'.repeat(64)},t);if(records.childNodes.some(n=>n.tagName==='DETAILS'))throw Error('wrong version record shown');
 if(!records.childNodes.some(n=>n.textContent===t('searchRecordsUnavailable')))throw Error('missing record not localized');
 renderSearchRecords(records,null,t);if(records.childNodes.some(n=>n.tagName==='DETAILS'))throw Error('version switch retains record');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_new_survey_default_is_v2_without_changing_legacy_options():
    html = Path("src/litagent/static/flow-demo/index.html").read_text()
    assert '<option value="academic_v2" selected' in html
    assert (
        '<option value="academic_v1"' in html
        and '<option value="method_transparent_v1"' in html
    )
    assert 'id="search-records"' in html
