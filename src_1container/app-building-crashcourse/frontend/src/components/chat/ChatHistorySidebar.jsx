'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Add, History } from '@mui/icons-material';
import DataService from "../../services/DataService";
import { formatRelativeTime } from "../../services/Common";


export default function ChatHistorySidebar({
    chat_id,
    model
}) {
    // Component States
    const [chatHistory, setChatHistory] = useState([]);
    const router = useRouter();

    // Setup Component
    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await DataService.GetChats(model, 20);
                setChatHistory(response.data);
            } catch (error) {
                console.error('Error fetching chats:', error);
                setChatHistory([]); // Set empty array in case of error
            }
        };

        fetchData();
    }, []);

    // UI View
    return (
        <div className="sidebar">
        <div className="sidebarHeader">
            <h2>Chat History</h2>
            <button
                className="newChatButton"
                onClick={() => router.push('/chat?model=' + model)}
            >
                New Chat
            </button>
        </div>

            <div className="flex-1 overflow-y-auto">
                {chatHistory.map((chat) => (
                    <div
                        key={chat.chat_id}
                        onClick={() => router.push(`/chat?model=${model}&id=${chat.chat_id}`)}
                        className={`p-4 border-b border-gray-200 cursor-pointer hover:bg-gray-50
                          transition-colors ${chat_id === chat.chat_id ? 'bg-purple-50' : ''}`}
                    >
                        <div className="flex flex-col gap-1">
                            <span className="text-gray-800 text-sm line-clamp-2">
                                {chat.title}
                            </span>
                            <span className="text-gray-500 text-xs">
                                {formatRelativeTime(chat.dts)}
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}