import{c as s,a as t,e as r,f as u,d as o,b as l,F as g,h as k,u as b,n as v,g as y,r as p,o as d,t as f,i as M,j as w,k as a}from"./app-lhHPgwf_.js";const T={__name:"page.html",setup(x){const m=M({setup(){const i=w("Hello world!"),e=n=>{i.value=n.target.value};return()=>[a("p",[a("span","输入: "),a("input",{value:i.value,onInput:e})]),a("p",[a("span","输出: "),i.value])]}});return(i,e)=>{const n=p("Badge"),c=p("VPCard");return d(),s("div",null,[e[2]||(e[2]=t("p",null,[t("code",null,"more"),l(" 注释之前的内容被视为文章摘要。")],-1)),r(" more "),e[3]||(e[3]=u(`<h2 id="页面标题" tabindex="-1"><a class="header-anchor" href="#页面标题"><span>页面标题</span></a></h2><p>The first H1 title in Markdown will be regarded as page title.</p><p>Markdown 中的第一个 H1 标题会被视为页面标题。</p><p>你可以在 Markdown 的 Frontmatter 中设置页面标题。</p><div class="language-md line-numbers-mode" data-highlighter="shiki" data-ext="md" style="--shiki-light:#383A42;--shiki-dark:#abb2bf;--shiki-light-bg:#FAFAFA;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes one-light one-dark-pro vp-code"><code><span class="line"><span style="--shiki-light:#383A42;--shiki-dark:#ABB2BF;">---</span></span>
<span class="line"><span style="--shiki-light:#383A42;--shiki-dark:#ABB2BF;">title: 页面标题</span></span>
<span class="line"><span style="--shiki-light:#383A42;--shiki-dark:#E06C75;">---</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="页面信息" tabindex="-1"><a class="header-anchor" href="#页面信息"><span>页面信息</span></a></h2><p>你可以在 Markdown 的 Frontmatter 中设置页面信息。</p><ul><li>作者设置为 Ms.Hope。</li><li>写作日期为 2020 年 1 月 1 日</li><li>分类为 “使用指南”</li><li>标签为 “页面配置” 和 “使用指南”</li></ul><h2 id="页面内容" tabindex="-1"><a class="header-anchor" href="#页面内容"><span>页面内容</span></a></h2><p>你可以自由在这里书写你的 Markdown。</p><div class="hint-container tip"><p class="hint-container-title">图片引入</p><ul><li>你可以将图片和 Markdown 文件放置在一起使用相对路径进行引用。</li><li>对于 <code>.vuepress/public</code> 文件夹的图片，请使用绝对链接 <code>/</code> 进行引用。</li></ul></div><h2 id="组件" tabindex="-1"><a class="header-anchor" href="#组件"><span>组件</span></a></h2><p>每个 Markdown 页面都会被转换为一个 Vue 组件，这意味着你可以在 Markdown 中使用 Vue 语法：</p><p>2</p>`,14)),r(" markdownlint-disable MD033 "),t("ul",null,[(d(),s(g,null,k(3,h=>t("li",null,f(h),1)),64))]),r(" markdownlint-enable MD033 "),e[4]||(e[4]=t("p",null,"你也可以创建并引入你自己的组件。",-1)),o(b(m)),e[5]||(e[5]=t("hr",null,null,-1)),e[6]||(e[6]=t("p",null,"主题包含一些有用的组件。这里是一些例子:",-1)),t("ul",null,[t("li",null,[t("p",null,[e[0]||(e[0]=l("文字结尾应该有深蓝色的 徽章文字 徽章。 ")),o(n,{text:"徽章文字",color:"#242378"})])]),t("li",null,[e[1]||(e[1]=t("p",null,"一个卡片:",-1)),o(c,v(y({title:"Mr.Hope",desc:"Where there is light, there is hope",logo:"https://mister-hope.com/logo.svg",link:"https://mister-hope.com",background:"rgba(253, 230, 138, 0.15)"})),null,16)])])])}}},j=JSON.parse('{"path":"/demo/page.html","title":"页面配置","lang":"zh-CN","frontmatter":{"title":"页面配置","cover":"/assets/images/cover1.jpg","icon":"file","order":3,"author":"Ms.Hope","date":"2020-01-01T00:00:00.000Z","category":["使用指南"],"tag":["页面配置","使用指南"],"sticky":true,"star":true,"footer":"这是测试显示的页脚","copyright":"无版权","description":"more 注释之前的内容被视为文章摘要。","head":[["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"页面配置\\",\\"image\\":[\\"https://mister-hope.github.io/assets/images/cover1.jpg\\"],\\"datePublished\\":\\"2020-01-01T00:00:00.000Z\\",\\"dateModified\\":\\"2025-04-16T08:15:52.000Z\\",\\"author\\":[{\\"@type\\":\\"Person\\",\\"name\\":\\"Ms.Hope\\"}]}"],["meta",{"property":"og:url","content":"https://mister-hope.github.io/demo/page.html"}],["meta",{"property":"og:site_name","content":"博客演示"}],["meta",{"property":"og:title","content":"页面配置"}],["meta",{"property":"og:description","content":"more 注释之前的内容被视为文章摘要。"}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:image","content":"https://mister-hope.github.io/assets/images/cover1.jpg"}],["meta",{"property":"og:locale","content":"zh-CN"}],["meta",{"property":"og:updated_time","content":"2025-04-16T08:15:52.000Z"}],["meta",{"name":"twitter:card","content":"summary_large_image"}],["meta",{"name":"twitter:image:src","content":"https://mister-hope.github.io/assets/images/cover1.jpg"}],["meta",{"name":"twitter:image:alt","content":"页面配置"}],["meta",{"property":"article:author","content":"Ms.Hope"}],["meta",{"property":"article:tag","content":"使用指南"}],["meta",{"property":"article:tag","content":"页面配置"}],["meta",{"property":"article:published_time","content":"2020-01-01T00:00:00.000Z"}],["meta",{"property":"article:modified_time","content":"2025-04-16T08:15:52.000Z"}]]},"git":{"createdTime":1744791352000,"updatedTime":1744791352000,"contributors":[{"name":"Pojis","username":"Pojis","email":"221220001@smail.nju.edu.cn","commits":1,"url":"https://github.com/Pojis"}]},"readingTime":{"minutes":1.76,"words":529},"filePathRelative":"demo/page.md","excerpt":"<p><code>more</code> 注释之前的内容被视为文章摘要。</p>\\n","autoDesc":true}');export{T as comp,j as data};
