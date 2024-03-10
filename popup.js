document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("news-form");
    const submitButton = document.getElementById("submit-button");
    const predictionResult = document.getElementById("prediction-result");
    const newsText = document.getElementById("news-text");
    const pic1 = document.getElementById("pic1");
    const pic2 = document.getElementById("pic2");
    const pic3 = document.getElementById("pic3");
    

  
    form.addEventListener("submit", async function (event) {
      event.preventDefault();

      const newsContent = newsText.value;
  
      const response = await fetch("http://127.0.0.1:5000/api/predict", {
        method: 'POST',
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ news: newsContent }),
      });
      console.log(response)
      const result = await response.json();
      displayResult(result);
    });
  
    function displayResult(result) {
      predictionResult.textContent = result.prediction || result.error;
  
      if (result.prediction === 'ham') {
        pic1.style.display = "none"
        pic2.style.display = "block"
        pic3.style.display = "none"
        predictionResult.textContent = "Reliable"
        predictionResult.style.color = "green";
        predictionResult.style.textAlign = "center";
        predictionResult.style.fontWeight = "bold";
      } else if (result.prediction === 'spam') {
        pic1.style.display = "none"
        pic2.style.display = "none  "
        pic3.style.display = "block"
        predictionResult.textContent = "Unreliable"
        predictionResult.style.color = "red";
        predictionResult.style.textAlign = "center";
        predictionResult.style.fontWeight = "bold";
      }else{
        pic1.style.display = "block"
        pic2.style.display = "none  "
        pic3.style.display = "none"
        predictionResult.style.color = "red";
        predictionResult.style.textAlign = "center";
        predictionResult.style.fontWeight = "bold";
      }
    }
  });
  