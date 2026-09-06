/* Palette toggle (default <-> slate); the initial palette is set by the inline script at the top of <body> */
(function () {
  var toggle = document.querySelector("[data-md-component=palette]");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = document.body.getAttribute("data-md-color-scheme") === "slate"
        ? "default" : "slate";
      document.body.setAttribute("data-md-color-scheme", next);
      try { localStorage.setItem("datadeer.palette", next); } catch (e) {}
    });
  }
})();

/* Highlight the current section in the table of contents */
(function () {
  var links = document.querySelectorAll(".md-nav--secondary a[href^='#']");
  if (!links.length || !("IntersectionObserver" in window)) return;
  var byId = {};
  links.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
  var current = null;
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      var link = byId[entry.target.id];
      if (!link) return;
      if (current) current.classList.remove("md-nav__link--active");
      link.classList.add("md-nav__link--active");
      current = link;
    });
  }, { rootMargin: "0px 0px -70% 0px" });
  Object.keys(byId).forEach(function (id) {
    var el = document.getElementById(id);
    if (el) observer.observe(el);
  });
})();
