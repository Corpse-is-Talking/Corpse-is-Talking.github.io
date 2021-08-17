---
title: "만화/아니메"
layout: archive
permalink: categories/AnimeManga
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.AnimeManga %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}