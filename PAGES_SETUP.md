# Acorn 1.0 on GitHub Pages

The Pages site in `docs/` is a public, read-only view. It does not host the Python dashboard, run the collector, train a model, show paper positions, or execute trades.

## Publish

1. Create or choose a GitHub repository. Review everything that will be public before pushing; a public repository exposes its committed source and history. This workflow lives in the bot project, so publishing that project from a public repository makes its committed source public too. To publish only the site, use a separate repository containing the `docs/` files and Pages workflow; keep the bot source private.
2. Push the project to the repository's `main` branch.
3. In repository **Settings → Pages**, set the build and deployment source to **GitHub Actions**.
4. The workflow in `.github/workflows/pages.yml` deploys the `docs/` directory after a push. GitHub Actions also supports running it manually from the Actions tab.

## Refresh the market snapshot

GitHub Pages serves files; it does not run this project's Python services. To publish a new data snapshot, run the local collector and then:

```powershell
python pages_export.py
```

Review `docs/status.json`, then commit and push that file. The exporter includes at most 25 latest per-token rows from DexScreener, and includes price rows only when the last scan is no more than three minutes old. It excludes paper trading history, model artifacts and predictions, logs, backups, account state, environment variables, and wallet data.

The website reports stale or unavailable data instead of presenting old prices as current. It is a point-in-time snapshot, not a live feed. Do not commit `.env`, `data/`, or `models/` to publish the site. If you deploy from a separate site-only repository, copy the generated `docs/status.json` into that repository's `docs/` folder before pushing.

See GitHub's [Pages quickstart](https://docs.github.com/en/pages/quickstart) and [custom Actions workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) for repository plan and Pages settings details.
