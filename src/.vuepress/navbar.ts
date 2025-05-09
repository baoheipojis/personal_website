import { navbar } from "vuepress-theme-hope";

export default navbar([
  "/",

  {
    text: "博文",
    icon: "pen-to-square",
    prefix: "/posts/",
    children: [
      {
        text: "个人认知与价值系统白皮书",
        icon: "pen-to-square",
        link:
          "/posts/杂项/个人认知与价值系统白皮书.html",
      },
      {
        text: "复现Attention Is All You Need",
        icon: "pen-to-square",
        link:
          "/posts/技术/llm/复现Attention Is All You Need.html",
      }

    ]
  },

]);
