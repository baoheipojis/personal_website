import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  "/": [
    "",
    {
      text: "如何使用",
      icon: "laptop-code",
      prefix: "demo/",
      link: "demo/",
      children: "structure",
    },
    {
      text: "文章",
      icon: "book",
      prefix: "posts/",
      // 'structure' should automatically pick up categories based on folders
      // or explicitly list them if needed.
      children: "structure",
      // Example of explicit listing if 'structure' doesn't work as desired:
      // children: [
      //   { text: "苹果", prefix: "apple/", children: "structure" },
      //   { text: "香蕉", prefix: "banana/", children: "structure" },
      //   "cherry",
      //   "dragonfruit",
      //   "strawberry",
      //   { text: "生活", link: "tomato" }, // Or prefix: "生活/" if it becomes a folder
      // ],
    },
    "intro",
    {
      text: "幻灯片",
      icon: "person-chalkboard",
      link: "https://ecosystem.vuejs.press/zh/plugins/markdown/revealjs/demo.html",
    },
  ],
});
