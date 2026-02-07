function getRecommendations() {
  let movie = document.getElementById("movie").value;
  document.getElementById("result").innerHTML =
    "Recommendations will appear here for <b>" + movie + "</b>";
}
