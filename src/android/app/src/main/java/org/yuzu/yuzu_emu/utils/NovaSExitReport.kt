// SPDX-FileCopyrightText: Copyright 2026 NovaS
// SPDX-License-Identifier: GPL-3.0-or-later

package org.yuzu.yuzu_emu.utils

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * NovaS: writes to eden_log.txt why the previous runs of the app ended, as Android recorded it.
 * A native crash on SIGSEGV or a kill by the low-memory killer leaves nothing in our own log (the
 * log just stops), so on the next start this asks Android and, for a native crash, prints the
 * crashing thread from the tombstone (signal, fault address, frames with library and offset).
 */
object NovaSExitReport {
    private const val MAX_EXITS = 3
    private const val MAX_FRAMES = 40

    fun log(context: Context) {
        Thread {
            try {
                report(context)
            } catch (e: Throwable) {
                Log.warning("[NovaSExit] could not read the previous exits: ${e.message}")
            }
        }.start()
    }

    private fun reasonName(reason: Int): String = when (reason) {
        ApplicationExitInfo.REASON_EXIT_SELF -> "exit-self"
        ApplicationExitInfo.REASON_SIGNALED -> "signaled"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "low-memory"
        ApplicationExitInfo.REASON_CRASH -> "java-crash"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "native-crash"
        ApplicationExitInfo.REASON_ANR -> "anr"
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "init-failure"
        ApplicationExitInfo.REASON_PERMISSION_CHANGE -> "permission-change"
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "excessive-resource-usage"
        ApplicationExitInfo.REASON_USER_REQUESTED -> "user-requested"
        ApplicationExitInfo.REASON_USER_STOPPED -> "user-stopped"
        ApplicationExitInfo.REASON_DEPENDENCY_DIED -> "dependency-died"
        ApplicationExitInfo.REASON_OTHER -> "other"
        ApplicationExitInfo.REASON_FREEZER -> "freezer"
        ApplicationExitInfo.REASON_PACKAGE_STATE_CHANGE -> "package-state-change"
        ApplicationExitInfo.REASON_PACKAGE_UPDATED -> "package-updated"
        else -> "reason-$reason"
    }

    private fun report(context: Context) {
        if (android.os.Build.VERSION.SDK_INT < android.os.Build.VERSION_CODES.R) return
        val am = context.getSystemService(ActivityManager::class.java) ?: return
        val exits = am.getHistoricalProcessExitReasons(context.packageName, 0, MAX_EXITS)
        if (exits.isEmpty()) return
        val time = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
        var tombstoneLogged = false
        for ((i, exit) in exits.withIndex()) {
            val mib = exit.pss / 1024
            val rss = exit.rss / 1024
            Log.info(
                "[NovaSExit] previous run ${i + 1}: ${time.format(Date(exit.timestamp))} " +
                    "reason=${reasonName(exit.reason)} status=${exit.status} " +
                    "importance=${exit.importance} pss=${mib}MiB rss=${rss}MiB " +
                    "description=${exit.description ?: "-"}"
            )
            // The newest native crash, even when an update or a stop came after it.
            if (!tombstoneLogged && exit.reason == ApplicationExitInfo.REASON_CRASH_NATIVE) {
                tombstoneLogged = true
                val bytes = try {
                    exit.traceInputStream?.use { it.readBytes() }
                } catch (e: Exception) {
                    null
                }
                if (bytes != null) logTombstone(bytes)
            }
        }
    }

    // --- minimal protobuf reader for Android's tombstone.proto (Android 12+) ---

    private class Field(val number: Int, val varint: Long, val bytes: ByteArray?)

    private fun fields(data: ByteArray): List<Field> {
        val out = ArrayList<Field>()
        var pos = 0
        fun varint(): Long {
            var result = 0L
            var shift = 0
            while (pos < data.size) {
                val b = data[pos++].toInt() and 0xff
                result = result or ((b and 0x7f).toLong() shl shift)
                if (b and 0x80 == 0) break
                shift += 7
            }
            return result
        }
        while (pos < data.size) {
            val key = varint()
            val number = (key ushr 3).toInt()
            when ((key and 7).toInt()) {
                0 -> out.add(Field(number, varint(), null))
                1 -> { pos += 8; out.add(Field(number, 0, null)) }
                2 -> {
                    val len = varint().toInt()
                    if (len < 0 || pos + len > data.size) return out
                    out.add(Field(number, 0, data.copyOfRange(pos, pos + len)))
                    pos += len
                }
                5 -> { pos += 4; out.add(Field(number, 0, null)) }
                else -> return out
            }
        }
        return out
    }

    private fun str(b: ByteArray?) = b?.toString(Charsets.UTF_8) ?: ""

    private fun logTombstone(data: ByteArray) {
        val top = fields(data)
        val tid = top.firstOrNull { it.number == 6 }?.varint ?: -1L
        // Signal (10): number 1, name 2, code 3, code_name 4, fault_address 9
        top.firstOrNull { it.number == 10 }?.bytes?.let { sig ->
            val f = fields(sig)
            Log.info(
                "[NovaSExit] signal ${f.firstOrNull { it.number == 1 }?.varint} " +
                    "${str(f.firstOrNull { it.number == 2 }?.bytes)} " +
                    "${str(f.firstOrNull { it.number == 4 }?.bytes)} " +
                    "fault address 0x${java.lang.Long.toHexString(f.firstOrNull { it.number == 9 }?.varint ?: 0)} " +
                    "crashing tid $tid"
            )
        }
        top.firstOrNull { it.number == 14 }?.bytes?.let { Log.info("[NovaSExit] abort message: ${str(it)}") }
        // Causes (15): human_readable 1
        top.filter { it.number == 15 }.forEach { c ->
            fields(c.bytes ?: return@forEach).firstOrNull { it.number == 1 }?.bytes?.let {
                Log.info("[NovaSExit] cause: ${str(it)}")
            }
        }
        // Threads (16): map<uint32, Thread>; Thread: id 1, name 2, registers 3, current_backtrace 4
        for (entry in top.filter { it.number == 16 }) {
            val kv = fields(entry.bytes ?: continue)
            if ((kv.firstOrNull { it.number == 1 }?.varint ?: -2L) != tid) continue
            val thread = fields(kv.firstOrNull { it.number == 2 }?.bytes ?: continue)
            Log.info("[NovaSExit] crashing thread '${str(thread.firstOrNull { it.number == 2 }?.bytes)}'")
            val regs = thread.filter { it.number == 3 }.mapNotNull { r ->
                val rf = fields(r.bytes ?: return@mapNotNull null)
                val name = str(rf.firstOrNull { it.number == 1 }?.bytes)
                if (name in setOf("pc", "lr", "sp", "x0", "x1", "x8", "x16", "x17", "x30")) {
                    "$name=0x${java.lang.Long.toHexString(rf.firstOrNull { it.number == 2 }?.varint ?: 0)}"
                } else {
                    null
                }
            }
            if (regs.isNotEmpty()) Log.info("[NovaSExit] registers ${regs.joinToString(" ")}")
            // BacktraceFrame: rel_pc 1, pc 2, function_name 4, function_offset 5, file_name 6, build_id 8
            thread.filter { it.number == 4 }.take(MAX_FRAMES).forEachIndexed { n, fr ->
                val ff = fields(fr.bytes ?: return@forEachIndexed)
                val rel = ff.firstOrNull { it.number == 1 }?.varint ?: 0
                val fn = str(ff.firstOrNull { it.number == 4 }?.bytes)
                val off = ff.firstOrNull { it.number == 5 }?.varint ?: 0
                val file = str(ff.firstOrNull { it.number == 6 }?.bytes)
                val sym = if (fn.isNotEmpty()) " ($fn+$off)" else ""
                Log.info("[NovaSExit] #%02d pc 0x%x %s%s".format(n, rel, file, sym))
            }
        }
    }
}
