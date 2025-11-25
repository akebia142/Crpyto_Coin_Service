console.log("😃regression_coin running!!!");
let anal_btn;
let ele_coin_name;
let timegap;
let res_contain;
let closeBtn;
let loader;
async function get_coinname() {
  ele_coin_name = document.getElementById("coinname");
  timegap = document.getElementById("timegap");
  res_contain = document.getElementById("res_contain");
  closeBtn = document.getElementById("closeBtn");
  loader = document.getElementById("loader");
  const conn = await fetch("/coin_name");
  const coinnames = await conn.json();
  let inHtml = "";
  for (let i = 0; i < coinnames.eng_name.length; i++) {
    inHtml += `<option value="${coinnames.eng_name[i]}">${coinnames.han_name[i]}(${coinnames.eng_name[i]})</option>`;
  }
  ele_coin_name.innerHTML = inHtml;
  //console.log(JSON.stringify(coinnames));
  anal_btn = document.getElementById("anal_btn");
  addEvent();
}

function addEvent() {
  closeBtn.addEventListener("click", async function () {
    res_contain.style.display = "none";
  });

  anal_btn.addEventListener("click", async function () {
    loader.style.display = "block";
    const style_loader = ` style="position:fixed;top:48vh;left:48vw"`;
    const loader_html = `<img id="loader_img" ${style_loader}src="/static/img/ajax-loader.gif">`;
    loader.innerHTML = loader_html;
    const coinname = ele_coin_name.value;
    const timegaps = timegap.value;
    const padding = await fetch("/user_data", {
      method: "post",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        coinname,
        timegaps,
      }),
    }).catch(() => {
      console.log("서버통신오류");
    });
    const info_data1 = await padding.json();
    const info_data = info_data1["report"]; //지금 안에 report로 감싸놨으니 그쪽으로 접근할수 있게해야함
    console.log(info_data);
    let inHtml2 = "";
    let today_date = new Date();
    today_date.setDate(today_date.getDate() + 1);
    let today_str = today_date.toLocaleString();
    let ghtml = `<h2 style="display:inline;padding:1rem;">
    ${coinname} 가격예측정보</h2>
    <p style="font-size:1.5rem;margin-bottom:1rem">최고가오차율(${
      info_data["err_rate"]["high"] * 100
    })%
    현재가오차율(${info_data["err_rate"]["cur"] * 100})%
    최저가오차율(${info_data["err_rate"]["low"] * 100})%
    </p>
    <div>
    <h2>훈련 결과 그래프</h2>
    <img style='width:20vw' src="/static/${info_data["graph"][0]}">
    <img style='width:20vw' src="/static/${info_data["graph"][1]}">
    </div>
    `;
    document.getElementById("anal_data").innerHTML = ghtml;
    for (let data of info_data["ypred"]) {
      //data[0] 현재가 data[1] 최고가 data[2] 최저가
      inHtml2 += `<div style='border:2px solid darkgray;margin-bottom:1rem'>
      <p style='background:green'>${today_str} </p>
      <p style='color:red'>최고가 : ${data[1]} </p>
      <p>현재가 : ${data[0]} </p>
      <p style='color:blue'>최저가 : ${data[2]} </p> 
      </div>
      `;
      document.getElementById("result").innerHTML = inHtml2;
      today_date.setDate(today_date.getDate() + 1);
      today_str = today_date.toLocaleString();
    }
    loader.style.display = "none";
    res_contain.style.display = "block";
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
