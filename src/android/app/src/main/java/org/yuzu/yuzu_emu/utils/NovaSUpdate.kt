// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

package org.yuzu.yuzu_emu.utils

import java.net.HttpURLConnection
import java.net.URL
import org.json.JSONArray
import org.yuzu.yuzu_emu.BuildConfig
import org.yuzu.yuzu_emu.NativeLibrary

/**
 * NovaS: looks for a newer Android release on GitHub (tags "android-X.Y.Z" in nova5emu/novas, the
 * same repository as the PC releases) and returns it with its NovaS-Android.apk, or null.
 */
object NovaSUpdate {
    private const val RELEASES = "https://api.github.com/repos/nova5emu/novas/releases?per_page=30"
    private const val ASSET = "NovaS-Android.apk"

    private fun parse(version: String): List<Int>? {
        val parts = version.trim().removePrefix("android-").removePrefix("v").split('.')
        if (parts.isEmpty() || parts.size > 4) return null
        return parts.map { it.toIntOrNull() ?: return null }
    }

    private fun newer(a: List<Int>, b: List<Int>): Boolean {
        for (i in 0 until maxOf(a.size, b.size)) {
            val x = a.getOrElse(i) { 0 }
            val y = b.getOrElse(i) { 0 }
            if (x != y) return x > y
        }
        return false
    }

    fun check(): NativeLibrary.UpdateResult? {
        val current = parse(BuildConfig.VERSION_NAME) ?: return null
        return try {
            val connection = URL(RELEASES).openConnection() as HttpURLConnection
            connection.connectTimeout = 8000
            connection.readTimeout = 8000
            connection.setRequestProperty("Accept", "application/vnd.github+json")
            val text = connection.inputStream.bufferedReader().use { it.readText() }
            connection.disconnect()
            val releases = JSONArray(text)
            var best: NativeLibrary.UpdateResult? = null
            var bestVersion: List<Int> = current
            for (i in 0 until releases.length()) {
                val release = releases.getJSONObject(i)
                val tag = release.optString("tag_name")
                if (!tag.startsWith("android-") || release.optBoolean("draft") ||
                    release.optBoolean("prerelease")
                ) {
                    continue
                }
                val version = parse(tag) ?: continue
                if (!newer(version, bestVersion)) continue
                val assets = release.optJSONArray("assets") ?: continue
                for (j in 0 until assets.length()) {
                    val asset = assets.getJSONObject(j)
                    if (asset.optString("name") == ASSET) {
                        best = NativeLibrary.UpdateResult(
                            tag = tag,
                            title = release.optString("name", tag),
                            body = release.optString("body"),
                            url = release.optString("html_url")
                        ).also { it.addAsset(asset.optString("browser_download_url")) }
                        bestVersion = version
                    }
                }
            }
            best
        } catch (e: Exception) {
            Log.warning("[NovaSUpdate] update check failed: ${e.message}")
            null
        }
    }
}
