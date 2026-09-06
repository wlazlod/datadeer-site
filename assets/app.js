/* Palette toggle (default <-> slate), persisted like mkdocs-material does */
(function () {
  var stored = null;
  try { stored = localStorage.getItem("datadeer.palette"); } catch (e) {}
  if (stored === "slate" || stored === "default") {
    document.body.setAttribute("data-md-color-scheme", stored);
  }
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

/* Fill PyPI version badges at load time; badges stay hidden if the lookup fails */
(function () {
  var badges = document.querySelectorAll("code[data-pypi]");
  if (!badges.length || !("fetch" in window)) return;
  badges.forEach(function (el) {
    fetch("https://pypi.org/pypi/" + el.getAttribute("data-pypi") + "/json")
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (data) {
        el.textContent = "v" + data.info.version;
        el.removeAttribute("hidden");
      })
      .catch(function () {});
  });
})();
