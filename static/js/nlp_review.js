console.log("😃regression_coin running!!!");
let user_text, loader, commitbtn, resetbtn, res_contain;
async function startpage() {
  user_text = document.getElementById("user_text");
  loader = document.getElementById("loader");
  commitbtn = document.getElementById("commitbtn");
  resetbtn = document.getElementById("resetbtn");
  res_contain = document.getElementById("res_contain");
  add_Event();
}

function add_Event() {
  commitbtn.addEventListener("click", async () => {
    console.log(user_text.value.length);
    const send_text = user_text.value;
    if (send_text.length < 10) {
      alert("10글자 이상 입력하셔야 합니다.");
      return 0;
    }

    loader.style.display = "block";
    const style_loader = ` style="position:fixed;top:48vh;left:48vw"`;
    const loader_html = `<img id="loader_img" ${style_loader}src="/static/img/ajax-loader.gif">`;
    loader.innerHTML = loader_html;
    const conn = await fetch("/movie_review", {
      method: "post",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ usertext: send_text }),
    });
    const pred_res = await conn.json();
    print(pred_res);
    loader.style.display = "none";
  });
  resetbtn.addEventListener("click", () => {
    user_text.value = "";
  });
}

document.addEventListener("click", (e) => {
  //파동효과
  const bg = document.getElementById("bg");

  const ripple = document.createElement("span");
  ripple.classList.add("bg-ripple");

  const size = Math.max(window.innerWidth, window.innerHeight) * 0.3;
  ripple.style.width = ripple.style.height = size + "px";
  ripple.style.left = e.clientX - size / 2 + "px";
  ripple.style.top = e.clientY - size / 2 + "px";

  bg.appendChild(ripple);

  setTimeout(() => ripple.remove(), 900);
});
